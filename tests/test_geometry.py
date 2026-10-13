from blv.utils.geometry import BoundingBox, describe_position, horizontal_position, relative_distance


def test_horizontal_position_left():
    box = BoundingBox(x1=0, y1=0, x2=50, y2=50)  # center_x = 25, frame_width = 640
    assert horizontal_position(box, frame_width=640) == "left"


def test_horizontal_position_center():
    box = BoundingBox(x1=280, y1=0, x2=360, y2=50)  # center_x = 320
    assert horizontal_position(box, frame_width=640) == "center"


def test_horizontal_position_right():
    box = BoundingBox(x1=550, y1=0, x2=620, y2=50)  # center_x = 585
    assert horizontal_position(box, frame_width=640) == "right"


def test_relative_distance_very_close():
    box = BoundingBox(x1=0, y1=0, x2=100, y2=400)  # height=400, frame_height=480 -> 0.83
    assert relative_distance(box, frame_height=480) == "very close"


def test_relative_distance_far():
    box = BoundingBox(x1=0, y1=0, x2=100, y2=40)  # height=40, frame_height=480 -> 0.08
    assert relative_distance(box, frame_height=480) == "far"


def test_describe_position_center_ahead():
    box = BoundingBox(x1=280, y1=0, x2=360, y2=400)
    phrase = describe_position(box, frame_width=640, frame_height=480)
    assert phrase.endswith("ahead")


def test_describe_position_side():
    box = BoundingBox(x1=0, y1=0, x2=50, y2=100)
    phrase = describe_position(box, frame_width=640, frame_height=480)
    assert "on your left" in phrase
