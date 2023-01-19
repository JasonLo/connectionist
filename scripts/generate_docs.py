from pathlib import Path


def write_md(module_name: str, class_name: str, docs_root: Path) -> None:
    """Write documentation markdown files for a class."""

    target_path = docs_root / module_name / (class_name + ".md")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w") as f:
        print(f"# {class_name}", file=f)
        print("", file=f)
        print(f"::: connectionist.{module_name}.{class_name}", file=f)


def main() -> None:
    """Generate documentation markdown files for all selected classes."""

    contents = dict(
        data=["ToyOP"],
        layers=[
            "TimeAveragedDense",
            "MultiInputTimeAveraging",
            "TimeAveragedRNNCell",
            "TimeAveragedRNN",
            "PMSPCell",
            "PMSPLayer",
            "Spoke",
            "HNSCell",
            "HNSLayer",
        ],
        models=["PMSP", "HubAndSpokes"],
        losses=["MaskedBinaryCrossEntropy"],
        surgery=["copy_transplant", "SurgeryPlan", "make_recipient", "Surgeon"],
    )

    for module_name, classes in contents.items():
        for class_name in classes:
            write_md(module_name, class_name, docs_root=Path("/connectionist/docs"))


if __name__ == "__main__":
    main()
