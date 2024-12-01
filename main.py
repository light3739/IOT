import json
import logging
import os
import threading
import time
import wave
from datetime import datetime
from queue import Queue

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import pyaudio
import sounddevice as sd
import whisper

from cam import ObjectDetector
from subscriber import MQTT_TOPIC

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Отключаем логи от opencv
logging.getLogger('opencv').setLevel(logging.WARNING)

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', "p4cfdde2.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv('MQTT_PORT', "8883"))
MQTT_USERNAME = os.getenv('MQTT_USERNAME', "esp8266_user")
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', "esp8266_user")
MQTT_CERT_PATH = os.getenv('MQTT_CERT_PATH', "emqxsl-ca.crt")
# MQTT Topics
OBJECT_DETECTION_TOPIC = "object_detection"
SPEECH_TOPIC = "speech_text"


class AudioVideoProcessor:
    def __init__(self):

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set(ca_certs=MQTT_CERT_PATH)

        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            logging.info(f"Connected to MQTT broker {MQTT_BROKER}")
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise
        # Set MQTT callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect

        # Audio configuration
        self.sample_rate = 16000
        self.audio_queue = Queue()
        self.chunk_size = 1024
        self.silence_threshold = 0.01
        self.min_audio_length = 0.5
        self.speech_topic = "speech_text"

        # Проверяем аудио устройства
        self.check_audio_devices()

        ffmpeg_bin_dir = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin"

        if os.path.exists(ffmpeg_bin_dir):
            # Добавляем путь к ffmpeg в PATH
            os.environ["PATH"] = os.pathsep.join([ffmpeg_bin_dir, os.environ["PATH"]])
            # Устанавливаем полный путь к исполняемому файлу ffmpeg
            os.environ["FFMPEG_BINARY"] = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
            logging.info(f"Added ffmpeg directory to PATH: {ffmpeg_bin_dir}")
        else:
            logging.error(f"ffmpeg directory not found at {ffmpeg_bin_dir}")

        if not self.check_ffmpeg():
            raise Exception("ffmpeg is not installed or not working properly")

        self.model = whisper.load_model("base")
        self.audio_buffer = []
        self.buffer_duration = 3  # секунды для накопления аудио
        self.samples_per_buffer = int(self.sample_rate * self.buffer_duration)

        # Создаем временную директорию
        self.temp_dir = "temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.object_detector = ObjectDetector(self.mqtt_client)
        self.camera = None

    def initialize_camera(self):
        """Инициализация камеры"""
        try:
            self.camera = cv2.VideoCapture(0)  # или URL вашей камеры
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
            logging.info("Camera initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize camera: {e}")
            return False
        return True

    def check_audio_devices(self):
        """Расширенная проверка аудио устройств"""
        try:
            p = pyaudio.PyAudio()
            logging.info("\n=== Audio Device Test ===")

            # Получаем информацию о всех устройствах
            input_devices = []
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get('maxInputChannels') > 0:  # только устройства с входом
                    input_devices.append(dev)
                    logging.info(f"\nDevice {i}: {dev.get('name')}")
                    logging.info(f"  Input channels: {dev.get('maxInputChannels')}")
                    logging.info(f"  Sample rate: {dev.get('defaultSampleRate')}")

            if not input_devices:
                logging.error("No input devices found!")
                return False

            # Тестируем выбранное устройство
            default = p.get_default_input_device_info()
            logging.info(f"\nTesting default device: {default.get('name')}")

            # Открываем тестовый поток
            test_stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024,
                input_device_index=int(default['index'])
            )

            # Читаем данные для проверки
            logging.info("Recording 3 seconds of audio to test microphone...")
            frames = []
            for _ in range(0, int(16000 / 1024 * 3)):
                data = test_stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))

            # Анализируем данные
            audio_data = np.concatenate(frames)
            level = np.abs(audio_data).mean()
            logging.info(f"Audio level detected: {level:.6f}")

            if level < 0.0001:
                logging.warning("Very low audio level detected. Microphone might not be working properly.")
            else:
                logging.info("Microphone test successful!")

            # Закрываем тестовый поток
            test_stream.stop_stream()
            test_stream.close()
            p.terminate()

            return True

        except Exception as e:
            logging.error(f"Error testing audio devices: {e}")
            if 'p' in locals():
                p.terminate()
            return False

    def check_ffmpeg(self):
        """Проверка установки ffmpeg"""
        try:
            import subprocess
            # Используем raw строку для пути
            ffmpeg_path = r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe"

            if not os.path.exists(ffmpeg_path):
                logging.error(f"ffmpeg not found at {ffmpeg_path}")
                return False

            result = subprocess.run([ffmpeg_path, '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                logging.info("ffmpeg is installed and working")
                version_line = result.stdout.splitlines()[0] if result.stdout else "Unknown version"
                logging.info(f"ffmpeg version: {version_line}")
                return True
            else:
                logging.error("ffmpeg is not working properly")
                return False
        except Exception as e:
            logging.error(f"Error checking ffmpeg: {str(e)}")
            return False

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            client.subscribe([
                (MQTT_TOPIC, 0),
                (self.speech_topic, 0),
                (OBJECT_DETECTION_TOPIC, 0)
            ])
            logging.info(f"Subscribed to topics: {MQTT_TOPIC}, {self.speech_topic}, {OBJECT_DETECTION_TOPIC}")
        else:
            logging.error(f"Failed to connect to MQTT broker with code: {rc}")

    def on_disconnect(self, client, userdata, rc, properties=None, reasonCode=None):
        """Callback для отключения от MQTT с поддержкой версии 2.0"""
        logging.warning(f"Disconnected from MQTT broker with code: {rc}")
        if rc != 0:
            try:
                self.mqtt_client.reconnect()
                logging.info("Reconnected to MQTT broker")
            except Exception as e:
                logging.error(f"Failed to reconnect: {e}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            logging.info(f"Received message: {payload}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def cleanup(self):
        logging.info("Cleaning up...")
        try:
            # Удаляем временные файлы
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info("Removed temporary directory")
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {e}")

        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()

    def audio_callback_pyaudio(self, in_data, frame_count, time_info, status):
        """Базовый callback для аудио"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paAbort)

    def reduce_noise(self, audio_data):
        """Simple noise reduction"""
        try:
            # Calculate noise floor
            noise_floor = np.percentile(np.abs(audio_data), 10)
            # Apply soft threshold
            audio_data = np.where(np.abs(audio_data) < noise_floor * 1.5, 0, audio_data)
            return audio_data
        except Exception as e:
            logging.error(f"Error in noise reduction: {e}")
            return audio_data  # Return original data if error occurs

    def process_audio(self):
        """Real-time audio processing with speech recognition"""
        logging.info("Starting real-time audio processing...")

        audio_buffer = []
        silence_threshold = 0.005  # Уменьшаем с 0.015 до 0.005
        min_silence_duration = 0.3  # Уменьшаем с 0.5 до 0.3
        silence_counter = 0
        last_process_time = time.time()
        process_interval = 1.0  # Уменьшаем с 2.0 до 1.0 секунды

        def clean_temp_files():
            try:
                for file in os.listdir(self.temp_dir):
                    if file.startswith("temp_speech"):
                        os.remove(os.path.join(self.temp_dir, file))
            except Exception as e:
                logging.error(f"Error cleaning temp files: {e}")

        while True:
            try:
                current_time = time.time()

                # Проверяем состояние аудио потока каждые 30 секунд
                if current_time - last_process_time > 30:
                    logging.info("Audio processing is still running")
                    last_process_time = current_time

                # Получаем данные из очереди с таймаутом
                try:
                    data = self.audio_queue.get(timeout=1.0)
                except Queue.Empty:
                    continue

                # Очищаем старые временные файлы
                if current_time - last_process_time > 10:
                    clean_temp_files()

                data = self.reduce_noise(data)
                audio_buffer.extend(data)

                current_level = np.abs(data).mean()
                logging.debug(f"Current audio level: {current_level}")  # Добавляем отладочный вывод

                if current_level > silence_threshold:
                    silence_counter = 0
                    if len(audio_buffer) >= self.sample_rate * 2:  # 2 seconds of audio
                        if current_time - last_process_time >= process_interval:
                            try:
                                # Нормализация аудио
                                audio_data = np.array(audio_buffer)
                                audio_data = audio_data / np.max(np.abs(audio_data))

                                # Генерируем уникальное имя файла
                                temp_filename = os.path.join(
                                    self.temp_dir,
                                    f"temp_speech_{int(time.time())}.wav"
                                )

                                with wave.open(temp_filename, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(self.sample_rate)
                                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                                # Распознавание речи
                                result = self.model.transcribe(
                                    temp_filename,
                                    language='pl',
                                    fp16=False,
                                    verbose=True,  # Включаем подробный вывод
                                    temperature=0.3,
                                    compression_ratio_threshold=2.0,
                                    no_speech_threshold=0.4,
                                    condition_on_previous_text=True
                                )

                                text = result["text"].strip()
                                if text:
                                    logging.info(f"Recognized text: {text}")

                                    # Отправка слов
                                    words = text.split()
                                    for word in words:
                                        message = {
                                            "timestamp": datetime.now().isoformat(),
                                            "word": word,
                                            "audio_level": float(current_level)
                                        }
                                        self.mqtt_client.publish(
                                            self.speech_topic,
                                            json.dumps(message),
                                            qos=0
                                        )

                                    # Отправка полного текста
                                    full_message = {
                                        "timestamp": datetime.now().isoformat(),
                                        "text": text,
                                        "audio_level": float(current_level)
                                    }
                                    self.mqtt_client.publish(
                                        f"{self.speech_topic}/full",
                                        json.dumps(full_message),
                                        qos=0
                                    )

                                # Очищаем буфер только после успешной обработки
                                audio_buffer = []
                                last_process_time = current_time

                            except Exception as e:
                                logging.error(f"Error processing audio: {e}", exc_info=True)
                                # Сохраняем проблемный аудио файл для анализа
                                error_file = f"error_audio_{int(time.time())}.wav"
                                with wave.open(error_file, 'wb') as wf:
                                    wf.setnchannels(1)
                                    wf.setsampwidth(2)
                                    wf.setframerate(self.sample_rate)
                                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                else:
                    silence_counter += len(data) / self.sample_rate
                    if silence_counter >= min_silence_duration:
                        audio_buffer = []
                        silence_counter = 0

                # Предотвращаем переполнение буфера
                if len(audio_buffer) > self.sample_rate * 10:
                    audio_buffer = audio_buffer[-self.sample_rate * 5:]
                    logging.warning("Audio buffer truncated to prevent overflow")

            except Exception as e:
                logging.error(f"Error in audio processing loop: {e}", exc_info=True)
                time.sleep(0.1)

    def initialize_audio(self):
        try:
            # Выбираем устройство Windows WASAPI (обычно работает лучше)
            devices = sd.query_devices()
            wasapi_inputs = [
                d for d in devices
                if 'WASAPI' in d['name'] and d['max_input_channels'] > 0
            ]

            if wasapi_inputs:
                device_index = wasapi_inputs[0]['index']
                logging.info(f"Using WASAPI device: {wasapi_inputs[0]['name']}")
            else:
                # Fallback to default input
                device_index = sd.default.device[0]
                logging.info(f"Using default device: {sd.query_devices(device_index)['name']}")

            return device_index
        except Exception as e:
            logging.error(f"Error initializing audio: {e}")
            return None

    def test_microphone(self, duration=5, filename=None):
        """Тестирование микрофона с записью в файл"""
        try:
            logging.info(f"Starting microphone test recording for {duration} seconds...")

            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mic_test_{timestamp}.wav"

            p = pyaudio.PyAudio()

            # Находим работающий микрофон
            device_index = self.select_best_microphone()
            if device_index is None:
                raise Exception("No working microphone found")

            # Настраиваем запись
            CHUNK = 1024
            FORMAT = pyaudio.paFloat32
            CHANNELS = 1
            RATE = self.sample_rate

            # Открываем поток без callback
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=device_index
            )

            logging.info("Recording...")
            frames = []

            # Записываем аудио
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                if i % 10 == 0:
                    progress = (i / (RATE / CHUNK * duration)) * 100
                    logging.info(f"Recording progress: {progress:.1f}%")

            logging.info("Finished recording")
            stream.stop_stream()
            stream.close()

            # Сохраняем в файл
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            p.terminate()
            return True, filename

        except Exception as e:
            logging.error(f"Error testing microphone: {e}")
            if 'p' in locals():
                p.terminate()
            return False, None

    def select_best_microphone(self):
        """Выбор микрофона номер 2"""
        try:
            p = pyaudio.PyAudio()
            target_index = 2  # Фиксированный индекс микрофона

            # Выводим список всех устройств для информации
            print("\nAvailable microphones:")
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels') > 0:
                    print(f"{i}: {dev_info['name']}")

            # Проверяем существование устройства с индексом 2
            try:
                device_info = p.get_device_info_by_index(target_index)
                if device_info.get('maxInputChannels') > 0:
                    logging.info(f"\nUsing fixed device {target_index}: {device_info['name']}")
                    return target_index
                else:
                    logging.error(f"Device {target_index} does not have input channels")
                    return None
            except Exception as e:
                logging.error(f"Error accessing device {target_index}: {e}")
                return None

        except Exception as e:
            logging.error(f"Error in microphone selection: {e}")
            return None
        finally:
            if 'p' in locals():
                p.terminate()

    def run(self):
        """Запуск монитора"""
        try:
            logging.info("Starting temperature and speech monitor...")

            # Инициализация детектора объектов и камеры
            self.object_detector = ObjectDetector(self.mqtt_client)
            camera_url = f"http://192.168.127.248:4747/video"  # URL для DroidCam
            self.camera = cv2.VideoCapture(camera_url)

            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
            logging.info("Camera initialized successfully")

            # Тестируем микрофон с записью
            success, test_file = self.test_microphone(duration=5)
            if not success:
                raise Exception("Microphone test failed!")

            logging.info(f"Microphone test recording saved to: {test_file}")
            logging.info("Please check the recorded file to verify microphone quality.")

            # Спрашиваем пользователя
            response = input("Did you check the recording? Press Enter to continue or 'n' to exit: ")
            if response.lower() == 'n':
                logging.info("Exiting as per user request")
                return

            # Initialize PyAudio with the tested device
            p = pyaudio.PyAudio()
            device_index = self.select_best_microphone()
            if device_index != 2:  # Проверяем, что используется нужный микрофон
                raise Exception("Failed to select microphone #2")

            device_info = p.get_device_info_by_index(device_index)
            logging.info(f"Using audio device: {device_info['name']}")

            # Start audio stream
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback_pyaudio,
                input_device_index=device_index
            )

            if not stream.is_active():
                raise Exception("Failed to start audio stream")

            stream.start_stream()
            logging.info("Audio stream started successfully")

            # Start audio processing thread
            audio_thread = threading.Thread(target=self.process_audio)
            audio_thread.daemon = True
            audio_thread.start()
            logging.info("Audio processing thread started")

            # Start video processing thread
            def video_processing_loop():
                last_image_time = time.time()
                image_interval = 3.0  # Интервал отправки изображений в секундах

                while True:
                    try:
                        ret, frame = self.camera.read()
                        current_time = time.time()

                        if ret:
                            processed_frame, fps, detected_objects = self.object_detector.process_frame(frame)

                            # Отправляем данные только каждые 3 секунды
                            if current_time - last_image_time >= image_interval:
                                if detected_objects:
                                    message = {
                                        "timestamp": datetime.now().isoformat(),
                                        "objects": detected_objects,
                                        "fps": fps
                                    }
                                    self.mqtt_client.publish(
                                        OBJECT_DETECTION_TOPIC,
                                        json.dumps(message),
                                        qos=0
                                    )
                                    last_image_time = current_time
                                    logging.info("Image data sent to broker")  # Добавляем лог для отслеживания отправки

                            cv2.imshow("Object Detection", processed_frame)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        # Небольшая задержка для снижения нагрузки на CPU
                        time.sleep(0.01)

                    except Exception as e:
                        logging.error(f"Error in video processing: {e}")
                        time.sleep(1)

            video_thread = threading.Thread(target=video_processing_loop)
            video_thread.daemon = True
            video_thread.start()
            logging.info("Video processing thread started")

            # Main loop
            while True:
                time.sleep(10)
                logging.info("System is running...")

        except KeyboardInterrupt:
            logging.info("Stopping monitor...")
        except Exception as e:
            logging.error(f"Error in run: {e}")
            raise
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'p' in locals():
                p.terminate()
            if hasattr(self, 'camera'):
                self.camera.release()
            cv2.destroyAllWindows()


def get_audio_device():
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        logging.info(f"Available audio devices: {devices}")
        logging.info(f"Default input device: {default_input}")
        return default_input['index']
    except Exception as e:
        logging.error(f"Error querying audio devices: {e}")
        return None


def main():
    processor = AudioVideoProcessor()
    try:
        # Удалить processor.connect()
        processor.run()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise
    finally:
        processor.cleanup()


if __name__ == "__main__":
    processor = AudioVideoProcessor()
    try:
        success, test_file = processor.test_microphone(duration=5)
        if success:
            logging.info(f"Test recording saved to: {test_file}")
            input("Press Enter after checking the recording...")

        processor.run()  # MQTT подключение уже выполнено в __init__
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        processor.cleanup()
