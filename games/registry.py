"""Game Component Registry — Genesis 2.0 Track EVA 02.

Registers and constructs game-specific component bundles implementing
the core protocols (StateParser, FeatureExtractor, SubgoalArbiter,
HypothesisBuilder, GameMemory, GameClient, DecisionPolicy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.protocols import (
    DecisionPolicy,
    FeatureExtractor,
    GameClient,
    GameMemory,
    HypothesisBuilder,
    StateParser,
    SubgoalArbiter,
)
from games.snake.client import SnakeClientWrapper
from games.snake.features import SnakeFeatureExtractor
from games.snake.hypotheses import SnakeHypothesisBuilder
from games.snake.memory import SnakeMemory
from games.snake.parser import SnakeStateParser
from games.snake.policy import SnakeDecisionPolicy
from games.snake.subgoals import SnakeSubgoalArbiter

from games.game2048.client import Game2048ClientWrapper
from games.game2048.features import Game2048FeatureExtractor
from games.game2048.hypotheses import Game2048HypothesisBuilder
from games.game2048.memory import Game2048Memory
from games.game2048.parser import Game2048StateParser
from games.game2048.policy import Game2048DecisionPolicy
from games.game2048.subgoals import Game2048SubgoalArbiter


@dataclass
class GameBundle:
    """Bundle of modular components required to run an autonomous game loop."""
    client: GameClient
    parser: StateParser
    features: FeatureExtractor
    subgoals: SubgoalArbiter
    hypotheses: HypothesisBuilder
    memory: GameMemory
    policy: DecisionPolicy
    legal_actions: list[str]
    max_steps: int


def get_game_bundle(
    game: str,
    api_key: str = "",
    agent_name: str = "ProgrammablePros",
    is_mock: bool = False,
) -> GameBundle:
    """Factory creating the complete component suite for a game."""
    game_clean = game.strip().lower()

    if game_clean == "snake":
        client = SnakeClientWrapper(api_key=api_key, agent_name=agent_name, is_mock=is_mock)
        parser = SnakeStateParser()
        memory = SnakeMemory()
        features = SnakeFeatureExtractor()
        subgoals = SnakeSubgoalArbiter()
        hypotheses = SnakeHypothesisBuilder()
        policy = SnakeDecisionPolicy()
        return GameBundle(
            client=client,
            parser=parser,
            features=features,
            subgoals=subgoals,
            hypotheses=hypotheses,
            memory=memory,
            policy=policy,
            legal_actions=["up", "down", "left", "right"],
            max_steps=300,
        )

    elif game_clean == "2048":
        client = Game2048ClientWrapper(api_key=api_key, agent_name=agent_name, is_mock=is_mock)
        parser = Game2048StateParser()
        memory = Game2048Memory()
        features = Game2048FeatureExtractor()
        subgoals = Game2048SubgoalArbiter()
        hypotheses = Game2048HypothesisBuilder()
        policy = Game2048DecisionPolicy()
        return GameBundle(
            client=client,
            parser=parser,
            features=features,
            subgoals=subgoals,
            hypotheses=hypotheses,
            memory=memory,
            policy=policy,
            legal_actions=["up", "down", "left", "right"],
            max_steps=2000,
        )

    raise ValueError(f"Unknown game '{game}'. Supported games are 'snake' and '2048'.")
