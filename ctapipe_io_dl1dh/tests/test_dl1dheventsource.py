## These tests should be enabled when we have a small test file

# example_file_path = 'output_file_1.h5'

# def test_is_compatible():
#     from ctapipe_io_dl1dh import DL1DHEventSource
#     assert DL1DHEventSource.is_compatible(example_file_path)
#
#
# def test_factory_for_dl1dh_file():
#     from ctapipe.io import event_source
#
#     reader = event_source(example_file_path)
#
#     # import here to see if ctapipe detects plugin
#     from ctapipe_io_dl1dh import DL1DHEventSource
#
#     assert isinstance(reader, DL1DHEventSource)
#     assert reader.input_url == example_file_path
