import json
from pathlib import Path


DATA_PATH = Path("data/raw/demo_sft.jsonl")


def main():

    samples = []

    with DATA_PATH.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            samples.append(
                json.loads(line)
            )

    print("Number of samples:")
    print(len(samples))

    print("\nFirst sample:")
    print(samples[0])

    print("\nMessages:")

    for message in samples[0]["messages"]:
        print(
            message["role"],
            "->",
            message["content"]
        )


if __name__ == "__main__":
    main()