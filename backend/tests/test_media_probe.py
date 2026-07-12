from salience_api.clips.media_probe import parse_ffprobe


def test_parse_ffprobe_extracts_video_metadata():
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "60/1",
            }
        ],
        "format": {
            "duration": "28.5",
            "size": "123456",
        },
    }

    metadata = parse_ffprobe(payload)

    assert metadata.duration_sec == 28.5
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == 60
    assert metadata.size_bytes == 123456
