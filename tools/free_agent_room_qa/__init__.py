"""Answer scoped floor-plan questions from original-page evidence on a CPU."""
from .answers import answer, answer_program, compile_question
from .reader import build_scene

__all__ = ['answer', 'answer_program', 'build_scene', 'compile_question']
