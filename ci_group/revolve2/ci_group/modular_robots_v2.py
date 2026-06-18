"""Standard modular robots."""

import numpy as np

from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2


def all() -> list[BodyV2]:
    """
    Get a list of all standard v2 modular robots.

    :returns: The list of robots.
    """
    return [
        babya_v2(),
        babyb_v2(),
        blokky_v2(),
        garrix_v2(),
        gecko_v2(),
        insect_v2(),
        linkin_v2(),
        longleg_v2(),
        penguin_v2(),
        pentapod_v2(),
        queen_v2(),
        salamander_v2(),
        squarish_v2(),
        snake_v2(),
        spider_v2(),
        stingray_v2(),
        tinlicker_v2(),
        turtle_v2(),
        ww_v2(),
        zappa_v2(),
        ant_v2(),
        park_v2(),
    ]


def spider_v2() -> BodyV2:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(0.0)

    return body


def gecko_v2() -> BodyV2:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def babya_v2() -> BodyV2:
    """
    Get the babya modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def ant_v2() -> BodyV2:
    """
    Get the ant modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.right.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def salamander_v2() -> BodyV2:
    """
    Get the salamander modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment = BrickV2(-np.pi / 2.0)

    body.core_v2.back_face.bottom.attachment.front.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.left = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.front.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )

    body.core_v2.back_face.bottom.attachment.front.front.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.left = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment = BrickV2(
        -np.pi / 2.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment.left = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def blokky_v2() -> BodyV2:
    """
    Get the blokky modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom = BrickV2(0.0)
    body.core_v2.back_face.bottom.right = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.front.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.front.attachment.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right.left = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front.front = ActiveHingeV2(0.0)

    return body


def park_v2() -> BodyV2:
    """
    Get the park modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.right = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.front = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.right = ActiveHingeV2(
        -np.pi / 2.0
    )
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.left = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.front.attachment = (
        BrickV2(0.0)
    )

    return body


def babyb_v2() -> BodyV2:
    """
    Get the babyb modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)

    return body


def garrix_v2() -> BodyV2:
    """
    Get the garrix modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.left = ActiveHingeV2(0.0)

    part2 = BrickV2(0.0)
    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.left = ActiveHingeV2(0.0)
    part2.left.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.left.attachment.attachment.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom.attachment.attachment.attachment.left.attachment = part2

    return body


def insect_v2() -> BodyV2:
    """
    Get the insect modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right.attachment = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right.attachment.attachment = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def linkin_v2() -> BodyV2:
    """
    Get the linkin modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)

    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment
    part2.front = BrickV2(0.0)

    part2.left = ActiveHingeV2(0.0)
    part2.left.attachment = ActiveHingeV2(0.0)

    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.right.attachment.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.attachment.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)

    return body


def longleg_v2() -> BodyV2:
    """
    Get the longleg modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment.attachment = BrickV2(
        0.0
    )

    part2 = body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment.attachment
    part2.right = ActiveHingeV2(0.0)
    part2.front = ActiveHingeV2(0.0)
    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.left.attachment.attachment = BrickV2(0.0)
    part2.left.attachment.attachment.right = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment.left.attachment = ActiveHingeV2(0.0)

    return body


def penguin_v2() -> BodyV2:
    """
    Get the penguin modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = BrickV2(0.0)
    body.core_v2.right_face.bottom.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment.attachment = BrickV2(
        -np.pi / 2.0
    )

    part2 = (
        body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment.attachment
    )

    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment = BrickV2(-np.pi / 2.0)

    part2.right = ActiveHingeV2(0.0)
    part2.right.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.attachment.attachment = BrickV2(-np.pi / 2.0)

    part2.right.attachment.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)

    part2.right.attachment.attachment.attachment.right = BrickV2(0.0)
    part2.right.attachment.attachment.attachment.right.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def pentapod_v2() -> BodyV2:
    """
    Get the pentapod modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment = BrickV2(-np.pi / 2.0)
    part2.front.attachment.left = BrickV2(0.0)
    part2.front.attachment.right = ActiveHingeV2(0.0)
    part2.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    part2.front.attachment.front.attachment.left = ActiveHingeV2(0.0)
    part2.front.attachment.front.attachment.right = ActiveHingeV2(0.0)

    return body


def queen_v2() -> BodyV2:
    """
    Get the queen modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.right = BrickV2(0.0)
    part2.right.front = BrickV2(0.0)
    part2.right.front.left = ActiveHingeV2(0.0)
    part2.right.front.right = ActiveHingeV2(0.0)

    part2.right.right = BrickV2(0.0)
    part2.right.right.front = ActiveHingeV2(np.pi / 2.0)
    part2.right.right.front.attachment = ActiveHingeV2(0.0)

    return body


def squarish_v2() -> BodyV2:
    """
    Get the squarish modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left.attachment.left = BrickV2(0.0)
    part2 = body.core_v2.back_face.bottom.attachment.left.attachment.left

    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.front = ActiveHingeV2(0.0)
    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment = BrickV2(-np.pi / 2.0)
    part2.right.attachment.left = BrickV2(0.0)
    part2.right.attachment.left.left = BrickV2(0.0)

    return body


def snake_v2() -> BodyV2:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment = (
        BrickV2(-np.pi / 2.0)
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        -np.pi / 2.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def stingray_v2() -> BodyV2:
    """
    Get the stingray modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.right = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.right = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.front = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.front.attachment = (
        BrickV2(0.0)
    )

    return body


def tinlicker_v2() -> BodyV2:
    """
    Get the tinlicker modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment

    part2.left = BrickV2(0.0)
    part2.left.front = ActiveHingeV2(np.pi / 2.0)
    part2.left.right = BrickV2(0.0)
    part2.left.right.left = BrickV2(0.0)
    part2.left.right.front = ActiveHingeV2(0.0)
    part2.left.right.front.attachment = BrickV2(0.0)
    part2.left.right.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    part2.left.right.front.attachment.right = BrickV2(0.0)
    part2.left.right.front.attachment.right.right = ActiveHingeV2(np.pi / 2.0)

    return body


def turtle_v2() -> BodyV2:
    """
    Get the turtle modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = BrickV2(0.0)
    body.core_v2.left_face.bottom.right = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom.left.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.right.attachment = BrickV2(0.0)
    part2 = body.core_v2.left_face.bottom.left.attachment.attachment.right.attachment

    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.front = BrickV2(0.0)
    part2.right = ActiveHingeV2(0.0)
    part2.right.attachment = BrickV2(0.0)
    part2.right.attachment.right = ActiveHingeV2(0.0)
    part2.right.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.right.attachment.left.attachment.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.left.attachment.attachment.attachment = ActiveHingeV2(0.0)

    return body


def ww_v2() -> BodyV2:
    """
    Get the ww modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.left.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.left.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.front = BrickV2(0.0)
    part2.front.right = ActiveHingeV2(np.pi / 2.0)
    part2.front.right.attachment = BrickV2(-np.pi / 2.0)
    part2.front.right.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.front.right.attachment.left.attachment = ActiveHingeV2(0.0)
    part2.front.right.attachment.left.attachment.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )

    return body


def zappa_v2() -> BodyV2:
    """
    Get the zappa modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment.attachment = BrickV2(
        0.0
    )
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment.attachment

    part2.front = ActiveHingeV2(0.0)
    part2.front.attachment = ActiveHingeV2(0.0)
    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = BrickV2(-np.pi / 2.0)
    part2.left.attachment.left = ActiveHingeV2(0.0)
    part2.left.attachment.left.attachment = BrickV2(0.0)
    part2.left.attachment.front = ActiveHingeV2(0.0)

    return body


def get(name: str) -> BodyV2:
    """
    Get a robot by name.

    :param name: The name of the robot to get.
    :returns: The robot with that name.
    :raises ValueError: When a robot with that name does not exist.
    """
    match name:
        case "gecko":
            return gecko_v2()
        case "spider":
            return spider_v2()
        case "babya":
            return babya_v2()
        case "ant":
            return ant_v2()
        case "salamander":
            return salamander_v2()
        case "blokky":
            return blokky_v2()
        case "park":
            return park_v2()
        case "babyb":
            return babyb_v2()
        case "garrix":
            return garrix_v2()
        case "insect":
            return insect_v2()
        case "linkin":
            return linkin_v2()
        case "longleg":
            return longleg_v2()
        case "penguin":
            return penguin_v2()
        case "pentapod":
            return pentapod_v2()
        case "queen":
            return queen_v2()
        case "squarish":
            return squarish_v2()
        case "snake":
            return snake_v2()
        case "stingray":
            return stingray_v2()
        case "tinlicker":
            return tinlicker_v2()
        case "turtle":
            return turtle_v2()
        case "ww":
            return ww_v2()
        case "zappa":
            return zappa_v2()
        case _:
            raise ValueError(f"Robot does not exist: {name}")
