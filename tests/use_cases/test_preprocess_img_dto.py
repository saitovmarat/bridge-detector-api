import pytest
from io import BytesIO
from PIL import Image
import numpy as np

# Импорты твоих функций
from bridges_detection_api.use_cases.img_preprocessor import preprocess_image_dto
from bridges_detection_api.infrastructure.img_processing import (
    array_to_dto, 
    image_to_array, 
    load_image_from_bytes
)
from bridges_detection_api.domain.image_dto import ImageDTO


def create_test_image_array(height=100, width=100):
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


@pytest.fixture
def test_image_stream():
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


# Тест: успешная обработка изображения
def test_preprocess_image_dto_success(test_image_stream, mocker):
    # Мокаем функции
    mock_load = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.load_image_from_bytes',
        wraps=load_image_from_bytes
    )
    mock_to_array = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.image_to_array',
        wraps=image_to_array
    )
    mock_to_dto = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.array_to_dto',
        wraps=array_to_dto
    )

    # Вызов тестируемого метода
    result = preprocess_image_dto(test_image_stream)

    # Проверки
    mock_load.assert_called_once_with(test_image_stream)
    mock_to_array.assert_called_once()
    mock_to_dto.assert_called_once()
    
    assert isinstance(result, ImageDTO)
    assert result.format == "BGR"


def test_preprocess_image_invalid_file(mocker):
    invalid_stream = BytesIO(b"not_an_image_data")
    mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.load_image_from_bytes',
        side_effect=ValueError("Cannot identify image file")
    )

    with pytest.raises(ValueError, match="Cannot identify image file"):
        preprocess_image_dto(invalid_stream)


def test_preprocess_image_function_call_order(test_image_stream, mocker):
    mock_load = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.load_image_from_bytes',
        return_value=Image.new('RGB', (10, 10))
    )
    mock_to_array = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.image_to_array',
        return_value=np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    )
    mock_to_dto = mocker.patch(
        'bridges_detection_api.use_cases.img_preprocessor.array_to_dto',
        autospec=True
    )

    preprocess_image_dto(test_image_stream)

    mock_load.assert_called_once()
    mock_to_array.assert_called_once_with(mock_load.return_value)
    mock_to_dto.assert_called_once_with(mock_to_array.return_value)