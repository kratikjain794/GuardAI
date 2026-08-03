from typing import Any

from bson import ObjectId


def serialize_document(document: dict[str, Any] | None):
    """
    Convert a MongoDB document into a JSON-friendly dictionary.
    ObjectId is converted to string.
    """

    if document is None:
        return None

    serialized = dict(document)

    if "_id" in serialized:
        serialized["_id"] = str(serialized["_id"])

    return serialized


def serialize_documents(documents: list[dict[str, Any]]):
    """Serialize multiple MongoDB documents."""

    return [
        serialize_document(document)
        for document in documents
    ]


def valid_object_id(value: str) -> bool:
    """Check whether a string is a valid MongoDB ObjectId."""

    return ObjectId.is_valid(value)


def object_id(value: str) -> ObjectId:
    """Convert string ID into MongoDB ObjectId."""

    if not valid_object_id(value):
        raise ValueError("Invalid MongoDB ObjectId")

    return ObjectId(value)