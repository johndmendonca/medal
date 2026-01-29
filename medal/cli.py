"""
Command-line interface for MEDAL framework.
"""
import click
from pathlib import Path
from typing import Optional

from medal.config import Config


@click.group()
@click.option(
    '--config',
    type=click.Path(exists=True),
    default='config.yaml',
    help='Path to configuration file'
)
@click.pass_context
def cli(ctx, config):
    """MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots."""
    ctx.ensure_object(dict)
    if Path(config).exists():
        ctx.obj['config'] = Config.from_yaml(config)
    else:
        ctx.obj['config'] = Config()


@cli.command()
@click.argument('dataset')
@click.option('--lang', required=True, help='Language code')
@click.option('--model', required=True, help='Model identifier')
@click.option('--run-id', default='vanilla', help='Run identifier')
@click.option('--type', type=click.Choice(['generate', 'evaluate', 'process']), required=True)
@click.pass_context
def narrative(ctx, dataset, lang, model, run_id, type):
    """Generate or evaluate narrative starters."""
    from medal.tasks.narrative_generation import NarrativeGenerator
    
    generator = NarrativeGenerator(
        dataset=dataset,
        lang=lang,
        model=model,
        run_id=run_id,
        config=ctx.obj['config']
    )
    
    if type == 'generate':
        generator.generate()
    elif type == 'evaluate':
        generator.evaluate()
    elif type == 'process':
        result = generator.regenerate()
        click.echo(f"Regenerating {result} examples.")


@cli.command()
@click.argument('context')
@click.option('--lang', required=True, help='Language code')
@click.option('--model', required=True, help='Model identifier')
@click.option('--role', type=click.Choice(['user', 'assistant']), required=True)
@click.option('--turn', type=int, required=True)
@click.option('--run-id', default='vanilla', help='Run identifier')
@click.option('--type', type=click.Choice(['generate', 'evaluate', 'process']), required=True)
@click.pass_context
def dialogue(ctx, context, lang, model, role, turn, run_id, type):
    """Generate or evaluate dialogue turns."""
    from medal.tasks.dialogue_generation import DialogueGenerator
    
    generator = DialogueGenerator(
        context=context,
        lang=lang,
        model=model,
        role=role,
        turn=turn,
        run_id=run_id,
        config=ctx.obj['config']
    )
    
    if type == 'generate':
        generator.generate()
    elif type == 'evaluate':
        generator.evaluate()
    elif type == 'process':
        result = generator.regenerate()
        click.echo(f"Regenerating {result} examples.")


@cli.command()
@click.argument('dialogue_path')
@click.option('--lang', required=True, help='Language code')
@click.option('--model', required=True, help='Model identifier')
@click.pass_context
def evaluate(ctx, dialogue_path, lang, model):
    """Evaluate complete dialogues."""
    from medal.tasks.dialogue_evaluation import DialogueEvaluator
    
    evaluator = DialogueEvaluator(
        dialogue_path=dialogue_path,
        lang=lang,
        model=model,
        config=ctx.obj['config']
    )
    evaluator.run()


def main():
    """Entry point for CLI."""
    cli()


if __name__ == '__main__':
    main()
