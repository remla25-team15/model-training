from unittest import mock

from src.get_data import (download_from_drive, download_nltk_resources,
                          run_get_data)


def test_download_from_drive_mocks_gdown():
    with mock.patch("src.get_data.gdown.download") as mock_gdown:
        file_id = "fake_id"
        output_path = "fake_path"
        download_from_drive(file_id, output_path)
        mock_gdown.assert_called_once()
        assert file_id in mock_gdown.call_args[0][0]


def test_download_nltk_resources_mocks_nltk():
    # Simulate wordnet found, should not call download
    with mock.patch("src.get_data.nltk.data.find", side_effect=LookupError), mock.patch(
        "src.get_data.nltk.download"
    ) as mock_download:
        result = download_nltk_resources()
        mock_download.assert_called_once_with("wordnet")
        assert result is True

    with mock.patch("src.get_data.nltk.data.find", return_value=True), mock.patch(
        "src.get_data.nltk.download"
    ) as mock_download:
        result = download_nltk_resources()
        mock_download.assert_not_called()
        assert result is False


def test_run_get_data_mocks_everything():
    with mock.patch("src.get_data.download_from_drive") as mock_drive, mock.patch(
        "src.get_data.download_nltk_resources", return_value=True
    ) as mock_nltk:
        file_ids = ["id1", "id2"]
        output_names = ["file1.txt", "file2.txt"]
        out_dir = "some_dir"
        result = run_get_data(
            output_dir=out_dir, file_ids=file_ids, output_names=output_names
        )
        assert result["nltk_downloaded"] is True
        assert result["downloaded"] == [f"{out_dir}/file1.txt", f"{out_dir}/file2.txt"]
        assert mock_drive.call_count == 2
        mock_nltk.assert_called_once()
