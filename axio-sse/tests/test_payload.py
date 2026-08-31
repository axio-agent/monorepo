"""What an event's data opens into, and how a handler reads it."""

import json
import logging

import pytest

from axio_sse import Event, MalformedPayload, Payload


async def test_a_broken_payload_is_reported_before_it_raises(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="axio.sse"), pytest.raises(MalformedPayload):
        Event(data="{not json").payload()
    assert [record.levelno for record in caplog.records] == [logging.ERROR]


@pytest.mark.parametrize("data", ["{not json", "[1,2]", "[DONE]"])
async def test_data_that_will_not_read_fails_the_stream(data: str) -> None:
    """Skipped instead, the events after it still finished the turn and reported success.

    A caller then got a short answer, or half a tool's arguments, with nothing saying that a
    delta had been dropped. A sentinel is data too: name it in `until` rather than let it through.
    """
    with pytest.raises(MalformedPayload):
        Event(data=data).payload()


async def test_an_event_with_no_data_carries_no_payload() -> None:
    # A name with no data is not corruption. The format allows it, and it reads as nothing.
    assert Event(data="").payload() is None


def test_a_path_that_is_missing_reads_as_the_default() -> None:
    payload = Payload({"message": {"usage": {"input_tokens": 7}}, "flag": True, "text": None})
    assert payload.number("message", "usage", "input_tokens") == 7
    assert payload.number("message", "usage", "output_tokens") == 0
    assert payload.number("message", "usage", "output_tokens", default=3) == 3
    assert payload.string("text") == ""
    assert payload.string("nothing", "deeper") == ""
    assert payload.obj("message", "usage") == {"input_tokens": 7}
    assert payload.obj("nothing") == {}
    assert payload.objs("nothing") == []


def test_a_true_flag_is_not_the_number_one() -> None:
    # bool is an int in Python, so a flag would otherwise read as a count and stay unnoticed.
    assert Payload({"flag": True}).number("flag") == 0


def test_a_payload_is_still_a_dict() -> None:
    payload = Payload({"a": 1})
    assert payload["a"] == 1 and "a" in payload and dict(payload) == {"a": 1}


def test_a_payload_can_still_be_dumped() -> None:
    # Google logs whole chunks with json.dumps, and redacts them with a helper that takes a dict.
    assert json.loads(json.dumps(Payload({"a": [1, 2]}))) == {"a": [1, 2]}
