from salience_media.probe import parse_ffprobe, safe_probe_media


def test_parse_ffprobe_reads_the_video_stream_and_format():
    metadata = parse_ffprobe(
        {
            "streams": [
                {"codec_type": "audio"},
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "60000/1001",
                },
            ],
            "format": {"duration": "30.5", "size": "1024"},
        }
    )

    assert (metadata.width, metadata.height) == (1920, 1080)
    assert metadata.duration_sec == 30.5
    assert metadata.size_bytes == 1024
    assert metadata.fps is not None and abs(metadata.fps - 59.94) < 0.01


def test_parse_ffprobe_tolerates_an_undefined_frame_rate():
    metadata = parse_ffprobe(
        {"streams": [{"codec_type": "video", "r_frame_rate": "0/0"}], "format": {}}
    )

    assert metadata.fps is None
    assert metadata.duration_sec is None


def test_safe_probe_falls_back_to_file_size_when_ffprobe_fails(tmp_path):
    clip = tmp_path / "not-a-video.mp4"
    clip.write_bytes(b"0123456789")

    metadata = safe_probe_media(clip)

    assert metadata.duration_sec is None
    assert metadata.size_bytes == 10
