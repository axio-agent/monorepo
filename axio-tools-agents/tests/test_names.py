import random
from collections import Counter

from axio_tools_agents.names import ADJECTIVES, SURNAMES, generate_name


def test_no_word_appears_twice() -> None:
    # Grace Hopper and Edward Hopper are two people and one string.
    for words in (ADJECTIVES, SURNAMES):
        assert [w for w, n in Counter(words).items() if n > 1] == []


def test_every_word_survives_being_an_id() -> None:
    for words in (ADJECTIVES, SURNAMES):
        for word in words:
            assert word.isascii() and word.isalpha() and word == word.lower(), word


def test_enough_pairs_that_a_collision_is_a_curiosity() -> None:
    assert len(ADJECTIVES) * len(SURNAMES) > 30_000


def test_a_name_is_two_words() -> None:
    adjective, _, surname = generate_name(random.Random(0)).partition("_")
    assert adjective in ADJECTIVES
    assert surname in SURNAMES


def test_the_same_seed_gives_the_same_name() -> None:
    assert generate_name(random.Random(7)) == generate_name(random.Random(7))
