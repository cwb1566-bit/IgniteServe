import argparse
import glob
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    total = 0
    for path in files:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            total += os.fstat(fd).st_size
        finally:
            os.close(fd)
    print(f"evicted_files={len(files)} evicted_bytes={total}", flush=True)


if __name__ == "__main__":
    main()
