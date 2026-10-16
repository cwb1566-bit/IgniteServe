# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Value-free input metadata serialization and reconstruction."""

import ast
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

from . import startup_assets as assets

logger = init_logger(__name__)


@dataclass
class TensorMetadata:
    """张量元信息（用于重建 FakeTensor）"""

    shape: List[Any]  # 可能包含符号维度（用字符串表示）
    dtype: str  # e.g., "torch.float16"
    device: str  # e.g., "cuda:0"
    stride: Optional[List[int]] = None
    requires_grad: bool = False
    is_sparse: bool = False
    storage_offset: int = 0
    layout: str = "torch.strided"

    def to_dict(self) -> Dict:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "shape": [str(s) for s in self.shape],  # 转换 SymInt 为字符串
            "dtype": self.dtype,
            "device": self.device,
            "stride": [str(s) for s in self.stride] if self.stride else None,
            "requires_grad": self.requires_grad,
            "is_sparse": self.is_sparse,
            "storage_offset": self.storage_offset,
            "layout": self.layout,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TensorMetadata":
        """从字典恢复（保持字符串格式，延迟解析）"""
        return TensorMetadata(
            # ========== 关键修复：不 eval，保持字符串 ==========
            # shape=[eval(s) if isinstance(s, str) else s for s in data['shape']],
            shape=data["shape"],
            dtype=data["dtype"],
            device=data["device"],
            # stride=[eval(s) if isinstance(s, str) else s for s in data['stride']] if data['stride'] else None,
            stride=data["stride"] if data["stride"] else None,
            requires_grad=data.get("requires_grad", False),
            is_sparse=data.get("is_sparse", False),
            storage_offset=data.get("storage_offset", 0),
            layout=data.get("layout", "torch.strided"),
        )


@dataclass
class SymIntMetadata:
    """SymInt 元信息"""

    expr: str  # 符号表达式，e.g., "s0", "s1 * 2"
    hint: Optional[int] = None  # 提示值（用于推导）

    def to_dict(self) -> Dict:
        return {
            "expr": self.expr,
            "hint": self.hint,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SymIntMetadata":
        return SymIntMetadata(
            expr=data["expr"],
            hint=data.get("hint"),
        )


@dataclass
class SubmoduleInputMetadata:
    """单个子模块的输入元信息"""

    submod_name: str
    index: int
    args_metadata: List[Any]  # TensorMetadata | SymIntMetadata | 原始值
    kwargs_metadata: Dict[str, Any]
    sym_shape_indices: List[int]
    graph_signature: Optional[str] = None
    contract: Optional[dict] = None

    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""

        def encode(value):
            if hasattr(value, "to_dict"):
                return value.to_dict()
            if isinstance(value, tuple):
                return {"__tuple__": [encode(v) for v in value]}
            if isinstance(value, list):
                return [encode(v) for v in value]
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            return value

        return {
            "submod_name": self.submod_name,
            "index": self.index,
            "args_metadata": [encode(arg) for arg in self.args_metadata],
            "kwargs_metadata": {k: encode(v) for k, v in self.kwargs_metadata.items()},
            "sym_shape_indices": self.sym_shape_indices,
            "graph_signature": self.graph_signature,
            "contract": self.contract,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SubmoduleInputMetadata":
        """从字典恢复"""

        def restore_arg(arg):
            if isinstance(arg, list):
                return [restore_arg(v) for v in arg]
            if isinstance(arg, dict):
                if "__tuple__" in arg:
                    return tuple(restore_arg(v) for v in arg["__tuple__"])
                if "shape" in arg and "dtype" in arg:
                    return TensorMetadata.from_dict(arg)
                elif "expr" in arg:
                    return SymIntMetadata.from_dict(arg)
                return {k: restore_arg(v) for k, v in arg.items()}
            return arg

        return SubmoduleInputMetadata(
            submod_name=data["submod_name"],
            index=data["index"],
            args_metadata=[restore_arg(arg) for arg in data["args_metadata"]],
            kwargs_metadata={
                k: restore_arg(v) for k, v in data["kwargs_metadata"].items()
            },
            sym_shape_indices=data["sym_shape_indices"],
            graph_signature=data.get("graph_signature"),
            contract=data.get("contract"),
        )


# ========== 序列化工具类 ==========
class SubmoduleInputSerializer:
    """子模块输入的序列化/反序列化工具"""

    def __init__(self, metadata_cache_dir: str, model_name, identity=None):
        path = Path(metadata_cache_dir)
        self.metadata_cache_dir = path.parent if path.suffix == ".json" else path
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = (
            path
            if path.suffix == ".json"
            else (path / f"submodule_inputs_metadata_for_{model_name}.json")
        )
        self.identity = {"schema": "submodule-inputs-v2", "runtime": identity}
        # Reuse the same symbolic object for the same textual expression so
        # reconstructed fake inputs preserve cross-argument shape relations.
        self.symbol_table: Dict[str, Any] = {}

    def seed_symbol_table_from_arg(self, arg: Any) -> None:
        """Seed symbolic dims from the current Dynamo/FakeTensor inputs."""
        if isinstance(arg, torch.Tensor):
            for dim in arg.shape:
                if isinstance(dim, torch.SymInt):
                    self.symbol_table.setdefault(str(dim), dim)
            try:
                for stride in arg.stride():
                    if isinstance(stride, torch.SymInt):
                        self.symbol_table.setdefault(str(stride), stride)
            except Exception:
                pass
            return
        if isinstance(arg, torch.SymInt):
            self.symbol_table.setdefault(str(arg), arg)
            return
        if isinstance(arg, (list, tuple)):
            for item in arg:
                self.seed_symbol_table_from_arg(item)
            return
        if isinstance(arg, dict):
            for item in arg.values():
                self.seed_symbol_table_from_arg(item)

    def seed_symbol_table_from_args(self, args: tuple[Any, ...]) -> None:
        for arg in args:
            self.seed_symbol_table_from_arg(arg)

    def extract_tensor_metadata(self, tensor: torch.Tensor) -> TensorMetadata:
        """从 Tensor 或 FakeTensor 提取元信息"""
        # 处理 SymInt 维度
        shape = []
        for s in tensor.shape:
            if isinstance(s, torch.SymInt):
                # 保存符号表达式
                shape.append(str(s))
            else:
                shape.append(int(s))

        # 处理 stride
        stride = None
        if hasattr(tensor, "stride"):
            stride = []
            for s in tensor.stride():
                if isinstance(s, torch.SymInt):
                    stride.append(str(s))
                else:
                    stride.append(int(s))

        return TensorMetadata(
            shape=shape,
            dtype=str(tensor.dtype),
            device=str(tensor.device),
            stride=stride,
            requires_grad=tensor.requires_grad,
            is_sparse=tensor.is_sparse,
            storage_offset=int(tensor.storage_offset()),
            layout=str(tensor.layout),
        )

    def extract_symint_metadata(self, sym_int: torch.SymInt) -> SymIntMetadata:
        """从 SymInt 提取元信息"""
        # 提取符号表达式
        expr = str(sym_int)

        # 尝试获取提示值
        hint = None
        if hasattr(sym_int, "node") and hasattr(sym_int.node, "hint"):
            hint = sym_int.node.hint

        return SymIntMetadata(expr=expr, hint=hint)

    def extract_arg_metadata(self, arg: Any) -> Any:
        """提取单个参数的元信息"""
        if isinstance(arg, torch.Tensor):
            return self.extract_tensor_metadata(arg)
        elif isinstance(arg, torch.SymInt):
            return self.extract_symint_metadata(arg)
        elif isinstance(arg, (list, tuple)):
            return type(arg)([self.extract_arg_metadata(a) for a in arg])
        elif isinstance(arg, dict):
            return {k: self.extract_arg_metadata(v) for k, v in arg.items()}
        else:
            # 原始类型（int, float, str, None 等）
            return arg

    def extract_submodule_metadata(
        self,
        submod_name: str,
        args: Tuple,
        kwargs: Dict,
        sym_shape_indices: List[int],
        index: int,
    ) -> SubmoduleInputMetadata:
        """提取子模块输入的元信息"""
        args_metadata = [self.extract_arg_metadata(arg) for arg in args]
        kwargs_metadata = {k: self.extract_arg_metadata(v) for k, v in kwargs.items()}

        return SubmoduleInputMetadata(
            submod_name=submod_name,
            index=index,
            args_metadata=args_metadata,
            kwargs_metadata=kwargs_metadata,
            sym_shape_indices=sym_shape_indices,
        )

    def save_all_metadata(self, metadata_list: List[SubmoduleInputMetadata]) -> None:
        """保存所有子模块的元信息到 JSON 文件"""
        data = {
            "version": "1.0",
            "num_submodules": len(metadata_list),
            "submodules": [meta.to_dict() for meta in metadata_list],
        }

        with assets.locked(self.metadata_file):
            assets.write_json(self.metadata_file, assets.record(self.identity, data))

        logger.info(
            f"Saved metadata for {len(metadata_list)} submodules to "
            f"{self.metadata_file}"
        )

    def load_all_metadata(self) -> List[SubmoduleInputMetadata]:
        """从 JSON 文件加载所有子模块的元信息"""
        if not self.metadata_file.exists():
            logger.warning(f"Metadata file not found: {self.metadata_file}")
            return []

        data = assets.read_record(self.metadata_file, self.identity)
        if data is None:
            return []

        metadata_list = [
            SubmoduleInputMetadata.from_dict(submod_data)
            for submod_data in data["submodules"]
        ]

        logger.info(
            f"Loaded metadata for {len(metadata_list)} submodules from "
            f"{self.metadata_file}"
        )
        return metadata_list

    def reconstruct_fake_tensor(
        self,
        metadata: TensorMetadata,
        fake_mode,
    ) -> torch.Tensor:
        """从元信息重建 FakeTensor"""
        # 解析 dtype
        dtype = getattr(torch, metadata.dtype.split(".")[-1])

        # 解析 device
        device = torch.device(metadata.device)

        # 解析 shape（可能包含 SymInt）
        shape = []
        for s in metadata.shape:
            if isinstance(s, str):
                # 检查是否是纯数字
                if s.isdigit():
                    shape.append(int(s))
                else:
                    # 这是一个符号表达式（如 "s0", "s1 * 2"）
                    # 创建新的 SymInt（使用默认提示值）
                    symint = self._create_symint(s, fake_mode)
                    shape.append(symint)
            else:
                shape.append(s)

        if (
            metadata.is_sparse
            or metadata.layout != "torch.strided"
            or metadata.storage_offset
        ):
            raise ValueError("unsupported layout or storage offset in cached input")
        if metadata.stride is None:
            raise ValueError("cached tensor has no stride")
        stride = [
            int(s) if str(s).isdigit() else self._create_symint(str(s), fake_mode)
            for s in metadata.stride
        ]
        # Preserve layout instead of silently rebuilding a contiguous tensor.
        with fake_mode:
            fake_tensor = torch.empty_strided(
                shape,
                stride,
                dtype=dtype,
                device=device,
                requires_grad=metadata.requires_grad,
            )

        return fake_tensor

    def _create_symint(
        self,
        expr: str,
        fake_mode,
        hint: Optional[int] = None,
    ) -> torch.SymInt:
        """
        创建 SymInt（从符号表达式）

        Args:
            expr: 符号表达式，如 "s0", "s1 * 2"
            fake_mode: FakeTensorMode
            hint: 提示值（默认从 expr 中提取或使用 1024）

        Returns:
            torch.SymInt
        """
        cached_symbol = self.symbol_table.get(expr)
        if cached_symbol is not None:
            return cached_symbol

        # Only expressions bound to the current Dynamo ShapeEnv are reusable.
        # Never manufacture a new unbacked symbol or substitute a hint.
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }

        def resolve(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return node.value
            if isinstance(node, ast.Name) and node.id in self.symbol_table:
                return self.symbol_table[node.id]
            if isinstance(node, ast.BinOp) and type(node.op) in operators:
                return operators[type(node.op)](resolve(node.left), resolve(node.right))
            raise ValueError(f"unbound or unsupported symbol: {expr}")

        value = resolve(ast.parse(expr, mode="eval").body)
        self.symbol_table[expr] = value
        return value

    def reconstruct_symint(
        self,
        metadata: SymIntMetadata,
        fake_mode,
    ) -> torch.SymInt:
        """从元信息重建 SymInt"""
        return self._create_symint(
            metadata.expr,
            fake_mode,
            hint=metadata.hint,
        )

    def reconstruct_arg(
        self,
        arg_metadata: Any,
        fake_mode,
    ) -> Any:
        """重建单个参数"""
        if isinstance(arg_metadata, TensorMetadata):
            return self.reconstruct_fake_tensor(arg_metadata, fake_mode)
        elif isinstance(arg_metadata, SymIntMetadata):
            return self.reconstruct_symint(arg_metadata, fake_mode)
        elif isinstance(arg_metadata, list):
            return [self.reconstruct_arg(a, fake_mode) for a in arg_metadata]
        elif isinstance(arg_metadata, tuple):
            return tuple([self.reconstruct_arg(a, fake_mode) for a in arg_metadata])
        elif isinstance(arg_metadata, dict):
            return {
                k: self.reconstruct_arg(v, fake_mode) for k, v in arg_metadata.items()
            }
        else:
            return arg_metadata

    def reconstruct_submodule_inputs(
        self,
        metadata: SubmoduleInputMetadata,
        fake_mode,
    ) -> Tuple[Tuple, Dict]:
        """重建子模块的输入"""
        args = tuple(
            [
                self.reconstruct_arg(arg_meta, fake_mode)
                for arg_meta in metadata.args_metadata
            ]
        )

        kwargs = {
            k: self.reconstruct_arg(v_meta, fake_mode)
            for k, v_meta in metadata.kwargs_metadata.items()
        }

        return args, kwargs
