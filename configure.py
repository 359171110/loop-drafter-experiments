"""Write a machine-local recipe without modifying the registered template."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--output', type=Path, default=Path('local/protocol.json'))
    args = parser.parse_args()
    recipe = json.loads((Path(__file__).parent / 'scripts/loop_drafter/data_protocol_v2.json').read_text())
    root = args.root.expanduser().resolve()
    recipe['runtime']['python'] = args.python
    recipe['capture']['target'] = str(args.model.expanduser().resolve())
    for section, key in [(recipe['training_data'], 'prompt_manifest'),
                         (recipe['training_data'], 'trajectory_manifest'),
                         (recipe['training'], 'frozen_io'),
                         (recipe['evaluation']['core'], 'manifest'),
                         (recipe['evaluation']['reasoning_extension'], 'manifest')]:
        section[key] = str(root / Path(section[key]).relative_to('/experiments/loop_drafter'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error(f'Refusing to overwrite {args.output}; choose another --output')
    args.output.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
