import os
import cv2
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# СТРУКТУРЫ ДАННЫХ
# ============================================================

@dataclass
class TextBlock:
    """Блок распознанного текста"""
    text: str
    confidence: float
    bbox: List[List[float]] = field(default_factory=list)
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False


@dataclass
class OCRResult:
    """Результат OCR обработки"""
    text: str
    blocks: List[TextBlock]
    confidence: float
    engine: str
    language: str
    metadata: Dict = field(default_factory=dict)


# ============================================================
# АБСТРАКТНЫЙ БАЗОВЫЙ КЛАСС OCR ДВИЖКА
# ============================================================

class OCREngine(ABC):
    """Абстрактный базовый класс для OCR движков"""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.name = "BaseEngine"
        self._is_available = False
        
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Распознавание текста на изображении"""
        pass
    
    @abstractmethod
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        """Распознавание с детальной информацией о блоках"""
        pass


# ============================================================
# ДВИЖОК 1: EASYOCR (рекомендуется для автономной работы)
# ============================================================

class EasyOCREngine(OCREngine):
    """
    EasyOCR - открытый код, PyTorch-based
    Поддержка 80+ языков, включая русский
    Не требует внешних бинарников
    """
    
    def __init__(self, languages: List[str] = None, gpu: bool = False):
        super().__init__(languages)
        self.name = "EasyOCR"
        self.gpu = gpu
        
        try:
            import easyocr
            # EasyOCR использует коды языков ISO 639-1
            lang_map = {
                'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'
            }
            self.easyocr_langs = [lang_map.get(l, l) for l in self.languages]
            
            logger.info(f"Инициализация EasyOCR с языками: {self.easyocr_langs}")
            self.reader = easyocr.Reader(
                self.easyocr_langs, 
                gpu=self.gpu,
                verbose=False
            )
            self._is_available = True
            logger.info("EasyOCR успешно инициализирован")
        except ImportError:
            logger.warning("EasyOCR не установлен. Установите: pip install easyocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"Ошибка инициализации EasyOCR: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Распознавание текста"""
        if not self._is_available:
            raise RuntimeError("EasyOCR недоступен")
        
        results = self.reader.readtext(image)
        
        blocks = []
        texts = []
        confidences = []
        
        for detection in results:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        """Распознавание с анализом размеров шрифтов"""
        result = self.recognize(image)
        
        # Анализ размеров шрифтов по bounding box
        for block in result.blocks:
            if block.bbox and len(block.bbox) >= 2:
                # Высота блока = приблизительный размер шрифта
                height = abs(block.bbox[2][1] - block.bbox[0][1])
                block.font_size = height
                
                # Эвристика для определения bold/italic
                # (по плотности пикселей в блоке)
                block.is_bold = self._estimate_bold(image, block.bbox)
        
        return result
    
    def _estimate_bold(self, image: np.ndarray, bbox: List) -> bool:
        """Эвристика для определения жирности шрифта"""
        try:
            x_coords = [int(p[0]) for p in bbox]
            y_coords = [int(p[1]) for p in bbox]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            if x_max <= x_min or y_max <= y_min:
                return False
            
            roi = image[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                return False
            
            # Конвертация в grayscale если нужно
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Бинаризация
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Плотность черных пикселей (жирный шрифт = больше пикселей)
            density = np.sum(binary > 0) / binary.size
            return density > 0.35
        except Exception:
            return False


# ============================================================
# ДВИЖОК 2: RAPIDOCR (легковесная, ONNX-based)
# ============================================================

class RapidOCREngine(OCREngine):
    """
    RapidOCR - легковесная обертка над PaddleOCR через ONNX
    Не требует PaddlePaddle, работает быстрее
    Открытый код: https://github.com/RapidAI/RapidOCR
    """
    
    def __init__(self, languages: List[str] = None):
        super().__init__(languages)
        self.name = "RapidOCR"
        
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.engine = RapidOCR()
            self._is_available = True
            logger.info("RapidOCR успешно инициализирован")
        except ImportError:
            logger.warning("RapidOCR не установлен. Установите: pip install rapidocr-onnxruntime")
            self._is_available = False
        except Exception as e:
            logger.error(f"Ошибка инициализации RapidOCR: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Распознавание текста"""
        if not self._is_available:
            raise RuntimeError("RapidOCR недоступен")
        
        result, _ = self.engine(image)
        
        if not result:
            return OCRResult(
                text='',
                blocks=[],
                confidence=0.0,
                engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for detection in result:
            bbox, text, confidence = detection
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        """Распознавание с деталями"""
        return self.recognize(image)


# ============================================================
# ДВИЖОК 3: PADDLEOCR (лучшее качество для русского)
# ============================================================

class PaddleOCREngine(OCREngine):
    """
    PaddleOCR от Baidu - открытый код
    Лучшее качество для русского языка
    Требует paddlepaddle + paddleocr
    """
    
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        super().__init__(languages)
        self.name = "PaddleOCR"
        
        try:
            from paddleocr import PaddleOCR
            
            # PaddleOCR использует коды 'ru', 'en'
            lang_map = {'ru': 'ru', 'en': 'en', 'rus': 'ru', 'eng': 'en'}
            primary_lang = lang_map.get(self.languages[0], 'ru')
            
            logger.info(f"Инициализация PaddleOCR с языком: {primary_lang}")
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=primary_lang,
                use_gpu=use_gpu,
                show_log=False
            )
            self._is_available = True
            logger.info("PaddleOCR успешно инициализирован")
        except ImportError:
            logger.warning("PaddleOCR не установлен. Установите: pip install paddlepaddle paddleocr")
            self._is_available = False
        except Exception as e:
            logger.error(f"Ошибка инициализации PaddleOCR: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Распознавание текста"""
        if not self._is_available:
            raise RuntimeError("PaddleOCR недоступен")
        
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return OCRResult(
                text='',
                blocks=[],
                confidence=0.0,
                engine=self.name,
                language='+'.join(self.languages)
            )
        
        blocks = []
        texts = []
        confidences = []
        
        for line in result[0]:
            bbox, (text, confidence) = line
            blocks.append(TextBlock(
                text=text,
                confidence=confidence,
                bbox=bbox
            ))
            texts.append(text)
            confidences.append(confidence)
        
        return OCRResult(
            text='\n'.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language='+'.join(self.languages)
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        """Распознавание с деталями"""
        return self.recognize(image)


# ============================================================
# ДВИЖОК 4: TESSERACT (для обратной совместимости)
# ============================================================

class TesseractEngine(OCREngine):
    """Классический Tesseract OCR"""
    
    def __init__(self, languages: List[str] = None, tesseract_path: str = None):
        super().__init__(languages)
        self.name = "Tesseract"
        
        try:
            import pytesseract
            
            # Настройка пути
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            elif os.name == 'nt':
                for path in [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                ]:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
            
            # Маппинг языков для Tesseract
            lang_map = {'ru': 'rus', 'en': 'eng', 'rus': 'rus', 'eng': 'eng'}
            self.tess_langs = '+'.join([lang_map.get(l, l) for l in self.languages])
            
            # Проверка доступности
            pytesseract.get_tesseract_version()
            self._is_available = True
            logger.info(f"Tesseract инициализирован с языками: {self.tess_langs}")
        except ImportError:
            logger.warning("pytesseract не установлен")
            self._is_available = False
        except Exception as e:
            logger.warning(f"Tesseract недоступен: {e}")
            self._is_available = False
    
    def recognize(self, image: np.ndarray) -> OCRResult:
        """Распознавание текста"""
        if not self._is_available:
            raise RuntimeError("Tesseract недоступен")
        
        import pytesseract
        
        text = pytesseract.image_to_string(
            image, 
            lang=self.tess_langs,
            config='--oem 3 --psm 6'
        )
        
        return OCRResult(
            text=text.strip(),
            blocks=[],
            confidence=0.0,
            engine=self.name,
            language=self.tess_langs
        )
    
    def recognize_with_details(self, image: np.ndarray) -> OCRResult:
        """Распознавание с детальными данными"""
        if not self._is_available:
            raise RuntimeError("Tesseract недоступен")
        
        import pytesseract
        
        data = pytesseract.image_to_data(
            image,
            lang=self.tess_langs,
            config='--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT
        )
        
        blocks = []
        texts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                block = TextBlock(
                    text=data['text'][i],
                    confidence=float(data['conf'][i]) / 100.0,
                    bbox=[
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ],
                    font_size=float(data['height'][i])
                )
                blocks.append(block)
                texts.append(data['text'][i])
                confidences.append(block.confidence)
        
        return OCRResult(
            text=' '.join(texts),
            blocks=blocks,
            confidence=float(np.mean(confidences)) if confidences else 0.0,
            engine=self.name,
            language=self.tess_langs
        )


# ============================================================
# МЕНЕДЖЕР OCR ДВИЖКОВ
# ============================================================

class OCREngineManager:
    """Менеджер для выбора и переключения OCR движков"""
    
    # Приоритет движков по качеству для русского языка
    PRIORITY = ['PaddleOCR', 'EasyOCR', 'RapidOCR', 'Tesseract']
    
    def __init__(self, preferred_engine: str = None, languages: List[str] = None):
        self.languages = languages or ['ru', 'en']
        self.engines: Dict[str, OCREngine] = {}
        self.current_engine: Optional[OCREngine] = None
        
        self._init_all_engines(preferred_engine)
    
    def _init_all_engines(self, preferred: str = None):
        """Инициализация всех доступных движков"""
        logger.info("Инициализация OCR движков...")
        
        # Инициализация всех движков
        engine_classes = [
            ('EasyOCR', lambda: EasyOCREngine(self.languages)),
            ('RapidOCR', lambda: RapidOCREngine(self.languages)),
            ('PaddleOCR', lambda: PaddleOCREngine(self.languages)),
            ('Tesseract', lambda: TesseractEngine(self.languages)),
        ]
        
        for name, factory in engine_classes:
            try:
                engine = factory()
                self.engines[name] = engine
                status = "✓ доступен" if engine.is_available else "✗ недоступен"
                logger.info(f"  {name}: {status}")
            except Exception as e:
                logger.warning(f"Ошибка инициализации {name}: {e}")
        
        # Выбор текущего движка
        if preferred and preferred in self.engines and self.engines[preferred].is_available:
            self.current_engine = self.engines[preferred]
        else:
            # Автоматический выбор по приоритету
            for engine_name in self.PRIORITY:
                if engine_name in self.engines and self.engines[engine_name].is_available:
                    self.current_engine = self.engines[engine_name]
                    break
        
        if self.current_engine:
            logger.info(f"Выбран движок: {self.current_engine.name}")
        else:
            logger.error("Ни один OCR движок не доступен!")
    
    def get_available_engines(self) -> List[str]:
        """Список доступных движков"""
        return [name for name, engine in self.engines.items() if engine.is_available]
    
    def switch_engine(self, engine_name: str) -> bool:
        """Переключение на другой движок"""
        if engine_name in self.engines and self.engines[engine_name].is_available:
            self.current_engine = self.engines[engine_name]
            logger.info(f"Переключено на: {engine_name}")
            return True
        return False
    
    def recognize(self, image: np.ndarray, with_details: bool = True) -> OCRResult:
        """Распознавание с текущим движком"""
        if not self.current_engine:
            raise RuntimeError("Нет доступных OCR движков")
        
        try:
            if with_details:
                return self.current_engine.recognize_with_details(image)
            else:
                return self.current_engine.recognize(image)
        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}")
            # Попытка использовать fallback движок
            return self._try_fallback(image, with_details)
    
    def _try_fallback(self, image: np.ndarray, with_details: bool) -> OCRResult:
        """Попытка использовать другой движок при ошибке"""
        for name, engine in self.engines.items():
            if engine.is_available and engine != self.current_engine:
                logger.info(f"Попытка использовать fallback: {name}")
                try:
                    if with_details:
                        return engine.recognize_with_details(image)
                    else:
                        return engine.recognize(image)
                except Exception:
                    continue
        raise RuntimeError("Все OCR движки не смогли обработать изображение")


# ============================================================
# ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ
# ============================================================

class ImagePreprocessor:
    """Предварительная обработка изображений для OCR"""
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Выравнивание наклона документа"""
        gray = ImagePreprocessor.to_grayscale(image)
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 100:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) < 0.5:
            return image
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), 
                              flags=cv2.INTER_CUBIC, 
                              borderMode=cv2.BORDER_REPLICATE)
    
    @classmethod
    def full_pipeline(cls, image: np.ndarray, 
                      deskew: bool = True,
                      denoise: bool = True,
                      enhance: bool = True,
                      binarize: bool = False) -> np.ndarray:
        """Полный пайплайн предобработки"""
        result = cls.to_grayscale(image)
        
        if deskew:
            result = cls.deskew(result)
        if denoise:
            result = cls.denoise(result)
        if enhance:
            result = cls.enhance_contrast(result)
        if binarize:
            result = cls.binarize(result)
        
        return result


# ============================================================
# АНАЛИЗАТОР ШРИФТОВ
# ============================================================

class FontAnalyzer:
    """Анализ шрифтов в документе"""
    
    FONT_PATTERNS = {
        'Serif': {'aspect_ratio': (0.4, 0.6), 'density': (0.25, 0.35)},
        'Sans-Serif': {'aspect_ratio': (0.45, 0.65), 'density': (0.30, 0.40)},
        'Monospace': {'aspect_ratio': (0.5, 0.7), 'density': (0.28, 0.38)},
    }
    
    @staticmethod
    def analyze_blocks(blocks: List[TextBlock]) -> Dict:
        """Анализ шрифтов по блокам"""
        if not blocks:
            return {'estimated_font': 'Unknown', 'average_size': 0}
        
        sizes = [b.font_size for b in blocks if b.font_size > 0]
        bold_count = sum(1 for b in blocks if b.is_bold)
        italic_count = sum(1 for b in blocks if b.is_italic)
        
        avg_size = float(np.mean(sizes)) if sizes else 0
        
        # Оценка типа шрифта
        font_type = 'Sans-Serif'  # по умолчанию
        
        # Эвристика по размеру
        if avg_size < 12:
            font_name = "Мелкий шрифт"
        elif avg_size < 16:
            font_name = "Стандартный (12-14pt)"
        elif avg_size < 20:
            font_name = "Крупный (14-18pt)"
        else:
            font_name = "Заголовочный"
        
        return {
            'estimated_font': font_name,
            'font_type': font_type,
            'average_size': avg_size,
            'size_range': [float(min(sizes)), float(max(sizes))] if sizes else [0, 0],
            'bold_percentage': (bold_count / len(blocks) * 100) if blocks else 0,
            'italic_percentage': (italic_count / len(blocks) * 100) if blocks else 0,
            'total_blocks': len(blocks)
        }


# ============================================================
# ИЗВЛЕЧЕНИЕ ТАБЛИЦ
# ============================================================

class TableExtractor:
    """Извлечение таблиц из документов"""
    
    @staticmethod
    def extract_from_pdf(pdf_path: str) -> List[Dict]:
        """Извлечение таблиц из PDF"""
        tables = []
        try:
            import camelot
            table_list = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(table_list):
                tables.append({
                    'table_id': i + 1,
                    'data': table.df.values.tolist(),
                    'accuracy': table.accuracy,
                    'page': table.page
                })
        except ImportError:
            logger.warning("Camelot не установлен. Установите: pip install camelot-py[cv]")
        except Exception as e:
            logger.error(f"Ошибка извлечения таблиц: {e}")
        
        return tables
    
    @staticmethod
    def extract_from_image(image: np.ndarray) -> List[Dict]:
        """Извлечение таблиц из изображения (эвристический метод)"""
        tables = []
        
        try:
            gray = ImagePreprocessor.to_grayscale(image)
            
            # Бинаризация
            _, binary = cv2.threshold(gray, 0, 255, 
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Детекция горизонтальных линий
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Детекция вертикальных линий
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            
            # Объединение
            grid = cv2.add(horizontal, vertical)
            
            # Поиск контуров
            contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            table_id = 1
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Таблица должна быть достаточно большой
                if w > 100 and h > 50:
                    tables.append({
                        'table_id': table_id,
                        'bbox': [x, y, w, h],
                        'note': 'Таблица обнаружена, используйте PDF для точного извлечения'
                    })
                    table_id += 1
        except Exception as e:
            logger.error(f"Ошибка детекции таблиц: {e}")
        
        return tables


# ============================================================
# ГЛАВНЫЙ ПРОЦЕССОР ДОКУМЕНТОВ
# ============================================================

class DocumentProcessor:
    """Главный процессор для обработки документов"""
    
    def __init__(self, 
                 ocr_engine: str = None,
                 languages: List[str] = None,
                 preprocess: bool = True):
        """
        Инициализация процессора
        
        Args:
            ocr_engine: Предпочтительный OCR движок (EasyOCR, RapidOCR, PaddleOCR, Tesseract)
            languages: Список языков ['ru', 'en']
            preprocess: Применять ли предобработку
        """
        self.languages = languages or ['ru', 'en']
        self.preprocess = preprocess
        
        # Инициализация менеджера OCR
        self.ocr_manager = OCREngineManager(
            preferred_engine=ocr_engine,
            languages=self.languages
        )
        
        self.preprocessor = ImagePreprocessor()
        self.font_analyzer = FontAnalyzer()
        self.table_extractor = TableExtractor()
    
    def process_image(self, image_path: str) -> Dict:
        """Обработка изображения"""
        logger.info(f"Обработка изображения: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {image_path}")
        
        # Предобработка
        if self.preprocess:
            processed = self.preprocessor.full_pipeline(image)
        else:
            processed = image
        
        # OCR распознавание
        ocr_result = self.ocr_manager.recognize(processed, with_details=True)
        
        # Анализ шрифтов
        font_analysis = self.font_analyzer.analyze_blocks(ocr_result.blocks)
        
        # Детекция таблиц
        tables = self.table_extractor.extract_from_image(image)
        
        return {
            'file': image_path,
            'type': 'image',
            'engine_used': ocr_result.engine,
            'text': ocr_result.text,
            'confidence': ocr_result.confidence,
            'language': ocr_result.language,
            'font_analysis': font_analysis,
            'tables': tables,
            'blocks_count': len(ocr_result.blocks),
            'metadata': {
                'image_size': image.shape[:2],
                'preprocessed': self.preprocess
            }
        }
    
    def process_pdf(self, pdf_path: str, dpi: int = 300) -> Dict:
        """Обработка PDF документа"""
        logger.info(f"Обработка PDF: {pdf_path}")
        
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("Установите: pip install pdf2image")
        
        images = convert_from_path(pdf_path, dpi=dpi)
        
        result = {
            'file': pdf_path,
            'type': 'pdf',
            'total_pages': len(images),
            'engine_used': self.ocr_manager.current_engine.name if self.ocr_manager.current_engine else 'None',
            'pages': [],
            'full_text': '',
            'tables': [],
            'overall_confidence': 0,
            'font_analysis': {}
        }
        
        all_blocks = []
        confidences = []
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Обработка страницы {page_num}/{len(images)}")
            
            img_array = np.array(image)
            
            # Предобработка
            if self.preprocess:
                processed = self.preprocessor.full_pipeline(img_array)
            else:
                processed = img_array
            
            # OCR
            ocr_result = self.ocr_manager.recognize(processed, with_details=True)
            
            page_data = {
                'page_number': page_num,
                'text': ocr_result.text,
                'confidence': ocr_result.confidence,
                'blocks_count': len(ocr_result.blocks)
            }
            
            result['pages'].append(page_data)
            result['full_text'] += f"\n--- Страница {page_num} ---\n{ocr_result.text}\n"
            all_blocks.extend(ocr_result.blocks)
            confidences.append(ocr_result.confidence)
        
        # Общая статистика
        result['overall_confidence'] = float(np.mean(confidences)) if confidences else 0
        result['font_analysis'] = self.font_analyzer.analyze_blocks(all_blocks)
        
        # Извлечение таблиц из PDF
        result['tables'] = self.table_extractor.extract_from_pdf(pdf_path)
        
        return result
    
    def process_document(self, file_path: str) -> Dict:
        """Универсальная обработка документа"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.webp']:
            return self.process_image(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    def save_results(self, result: Dict, output_path: str):
        """Сохранение результатов"""
        output_path = str(output_path)
        
        # JSON с полной информацией
        json_path = f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        # Текстовый файл
        txt_path = f"{output_path}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result.get('full_text', result.get('text', '')))
        
        # Таблицы в CSV
        if result.get('tables'):
            for i, table in enumerate(result['tables'], 1):
                if 'data' in table:
                    csv_path = f"{output_path}_table_{i}.csv"
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(table['data'])
        
        logger.info(f"Результаты сохранены: {output_path}.*")
        return {
            'json': json_path,
            'txt': txt_path,
            'tables_count': len(result.get('tables', []))
        }


# ============================================================
# ИНТЕРФЕЙС КОМАНДНОЙ СТРОКИ
# ============================================================

def print_banner():
    print("=" * 70)
    print("  ПРОГРАММА РАСПОЗНАВАНИЯ ДОКУМЕНТОВ")
    print("  Поддержка: Русский + Английский язык")
    print("  Движки: EasyOCR | RapidOCR | PaddleOCR | Tesseract")
    print("=" * 70)


def interactive_mode():
    """Интерактивный режим работы"""
    print_banner()
    
    # Создание процессора
    processor = DocumentProcessor(
        languages=['ru', 'en'],
        preprocess=True
    )
    
    # Информация о доступных движках
    available = processor.ocr_manager.get_available_engines()
    print(f"\n✓ Доступные OCR движки: {', '.join(available) if available else 'НИ ОДНОГО'}")
    
    if not available:
        print("\n⚠ Установите хотя бы один движок:")
        print("  pip install easyocr")
        print("  pip install rapidocr-onnxruntime")
        print("  pip install paddlepaddle paddleocr")
        return
    
    current = processor.ocr_manager.current_engine
    print(f"✓ Активный движок: {current.name}")
    
    while True:
        print("\n" + "-" * 70)
        print("1. Обработать документ")
        print("2. Сменить OCR движок")
        print("3. Показать доступные движки")
        print("4. Выход")
        
        choice = input("\nВыберите действие: ").strip()
        
        if choice == '1':
            file_path = input("Введите путь к файлу: ").strip().strip('"\'')
            if not file_path:
                print("Путь не указан")
                continue
            
            try:
                result = processor.process_document(file_path)
                
                print("\n" + "=" * 70)
                print("РЕЗУЛЬТАТЫ")
                print("=" * 70)
                print(f"Движок: {result['engine_used']}")
                print(f"Уверенность: {result.get('overall_confidence', result.get('confidence', 0)):.2%}")
                
                # Показ текста
                text = result.get('full_text', result.get('text', ''))
                preview = text[:500] + "..." if len(text) > 500 else text
                print(f"\nТекст (фрагмент):\n{preview}")
                
                # Анализ шрифтов
                if result.get('font_analysis'):
                    fa = result['font_analysis']
                    print(f"\nШрифты:")
                    print(f"  Тип: {fa.get('estimated_font', 'Unknown')}")
                    print(f"  Средний размер: {fa.get('average_size', 0):.1f} px")
                    print(f"  Жирный: {fa.get('bold_percentage', 0):.1f}%")
                
                # Таблицы
                if result.get('tables'):
                    print(f"\nТаблиц: {len(result['tables'])}")
                
                # Сохранение
                output = os.path.splitext(file_path)[0] + "_result"
                saved = processor.save_results(result, output)
                print(f"\n✓ Сохранено: {saved['json']}")
                
            except Exception as e:
                print(f"\n✗ Ошибка: {e}")
                logger.exception("Детали:")
        
        elif choice == '2':
            available = processor.ocr_manager.get_available_engines()
            print("\nДоступные движки:")
            for i, name in enumerate(available, 1):
                marker = " ← текущий" if name == processor.ocr_manager.current_engine.name else ""
                print(f"  {i}. {name}{marker}")
            
            try:
                idx = int(input("Выберите номер: ")) - 1
                if 0 <= idx < len(available):
                    if processor.ocr_manager.switch_engine(available[idx]):
                        print(f"✓ Переключено на: {available[idx]}")
            except (ValueError, IndexError):
                print("Неверный выбор")
        
        elif choice == '3':
            print("\nВсе движки:")
            for name, engine in processor.ocr_manager.engines.items():
                status = "✓" if engine.is_available else "✗"
                current = " ← текущий" if engine == processor.ocr_manager.current_engine else ""
                print(f"  {status} {name}{current}")
        
        elif choice == '4':
            print("До свидания!")
            break
        
        else:
            print("Неверный выбор")


def main():
    """Главная функция"""
    import sys
    
    if len(sys.argv) > 1:
        # Режим командной строки
        file_path = sys.argv[1]
        engine = sys.argv[2] if len(sys.argv) > 2 else None
        
        processor = DocumentProcessor(ocr_engine=engine, languages=['ru', 'en'])
        result = processor.process_document(file_path)
        output = os.path.splitext(file_path)[0] + "_result"
        processor.save_results(result, output)
        print(f"Готово. Результаты в {output}.*")
    else:
        # Интерактивный режим
        interactive_mode()


if __name__ == "__main__":
    main()