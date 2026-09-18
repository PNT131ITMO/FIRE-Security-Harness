import argparse
import dataclasses
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core.agent import Model
from src.core.pipeline import verify_atomic_claim
from src.core.types import VerificationError
from src.utils.config_loader import load_config, validate_baseline_config
from src.utils.text_utils import to_readable_json


def main() -> None:
    parser = argparse.ArgumentParser(description='FIRE atomic claim verification')
    parser.add_argument('claim', nargs='?',
                        help='One atomic claim to verify; prompts when omitted')
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'configs' / 'baseline.yaml')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--search-provider', choices=('serper', 'tavily'),
                        help='Override the search provider from YAML for this run')
    args = parser.parse_args()
    if args.claim is None:
        try:
            args.claim = input('Nhap atomic claim: ').strip()
        except EOFError:
            parser.error('No claim received. Pass a claim argument or enter one at the prompt.')
        except KeyboardInterrupt:
            raise SystemExit(130) from None
    if not args.claim.strip():
        parser.error('Claim must not be empty.')
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / '.env', override=False, encoding='utf-8-sig')

    config = load_config(args.config)
    if args.search_provider is not None:
        if not isinstance(config.get('search'), dict):
            raise ValueError("Configuration section 'search' must be a mapping.")
        config['search']['type'] = args.search_provider
    validate_baseline_config(config)
    model_options = dict(config['model'])
    model_options['model_name'] = model_options.pop('name')
    fire_options = dict(config['fire'])
    fire_options['tolerance'] = fire_options.pop('max_tolerance')
    try:
        answer, searches, usage = verify_atomic_claim(
            args.claim, Model(**model_options), **fire_options,
            search_type=config['search']['type'],
            num_searches=config['search']['num_searches'],
        )
    except VerificationError as error:
        print(to_readable_json({'claim': args.claim, 'result': None,
                               'error': str(error), 'searches': error.searches,
                               'usage': error.usage}))
        raise SystemExit(1) from error
    print(to_readable_json({'claim': args.claim,
                           'result': dataclasses.asdict(answer) if answer else None,
                           'searches': searches, 'usage': usage}))
    if answer is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
