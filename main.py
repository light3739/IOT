import os
import wave
from datetime import datetime

import cv2
import paho.mqtt.client as mqtt
import pyaudio
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import time

import logging
import whisper
import sounddevice as sd
import numpy as np
import threading
from queue import Queue

from cam import ObjectDetector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# MQTT Configuration
MQTT_BROKER = "p4cfdde2.ala.eu-central-1.emqxsl.com"
MQTT_PORT = 8883
MQTT_TOPIC = "test"
MQTT_USERNAME = "esp8266_user"
MQTT_PASSWORD = "esp8266_user"

# InfluxDB Configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "your-super-secret-auth-token"  # Тот же токен, что в docker-compose
INFLUXDB_ORG = "myorg"
INFLUXDB_BUCKET = "temperature_data"

# SSL Certificate path
CERT_PATH = "emqxsl-ca.crt"


class TemperatureAndSpeechMonitor:
    def __init__(self):
        # Базовая инициализация
        self.messages_received = 0
        self.messages_written = 0

        # Initialize InfluxDB client
        self.influx_client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # Используем VERSION2
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set(ca_certs=CERT_PATH)

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

    def connect(self):
        try:
            logging.info(f"Connecting to MQTT broker {MQTT_BROKER}...")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback для подключения к MQTT с поддержкой версии 2.0"""
        if rc == 0:
            logging.info("Connected to MQTT broker")
            client.subscribe([(MQTT_TOPIC, 0), (self.speech_topic, 0)])
            logging.info(f"Subscribed to topics: {MQTT_TOPIC}, {self.speech_topic}")
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

    def write_to_influxdb(self, point):
        try:
            self.write_api.write(bucket=INFLUXDB_BUCKET, record=point)
            self.messages_written += 1
            return True
        except Exception as e:
            logging.error(f"Failed to write to InfluxDB: {e}")
            return False

    def on_message(self, client, userdata, msg):
        try:
            # Parse the message
            payload = json.loads(msg.payload.decode())
            self.messages_received += 1
            logging.info(f"Received message: {payload}")

            # Обновленная логика для нового формата сообщений
            temperature = payload.get("temperature")
            if temperature is not None:
                # Create InfluxDB point
                point = Point("temperature") \
                    .tag("device", payload.get("device", "unknown")) \
                    .field("value", float(temperature)) \
                    .time(datetime.utcnow())

                # Write to InfluxDB
                if self.write_to_influxdb(point):
                    logging.info(f"Temperature {temperature}°C written to InfluxDB")
            else:
                logging.warning("No temperature value in message")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def print_stats(self):
        logging.info(f"\nStatistics:")
        logging.info(f"Messages received: {self.messages_received}")
        logging.info(f"Messages written to InfluxDB: {self.messages_written}")

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
        self.influx_client.close()

    def audio_callback_pyaudio(self, in_data, frame_count, time_info, status):
        """Базовый callback для аудио"""
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
            return (None, pyaudio.paAbort)

    def process_audio(self):
        """Real-time audio processing with speech recognition"""
        logging.info("Starting real-time audio processing...")

        # Initialize audio buffer
        audio_buffer = []
        silence_threshold = 0.03  # Increased from 0.01 to handle noise
        min_silence_duration = 1.0  # Increased from 0.5 to better detect speech gaps
        silence_counter = 0

        def reduce_noise(audio_data):
            """Simple noise reduction"""
            # Calculate noise floor
            noise_floor = np.percentile(np.abs(audio_data), 20)
            # Apply soft threshold
            audio_data = np.where(np.abs(audio_data) < noise_floor * 2, 0, audio_data)
            return audio_data

        while True:
            try:
                if not self.audio_queue.empty():
                    # Get data from queue
                    data = self.audio_queue.get()

                    data = reduce_noise(data)
                    audio_buffer.extend(data)

                    # Check audio level
                    current_level = np.abs(data).mean()

                    if current_level > silence_threshold:
                        silence_counter = 0
                        # If enough data, start processing
                        if len(audio_buffer) >= self.sample_rate * 2:  # 2 seconds of audio
                            # Normalize audio
                            audio_data = np.array(audio_buffer)
                            audio_data = audio_data / np.max(np.abs(audio_data))

                            # Save to temp file
                            temp_filename = os.path.join(self.temp_dir, "temp_speech.wav")
                            with wave.open(temp_filename, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(self.sample_rate)
                                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                            # Recognize speech
                            try:
                                result = self.model.transcribe(
                                    temp_filename,
                                    language='pl',
                                    fp16=False,
                                    verbose=False,
                                    # Additional Whisper parameters for noisy environment
                                    temperature=0.2,  # Lower temperature for more focused predictions
                                    compression_ratio_threshold=2.4,  # Adjust for better noise handling
                                    no_speech_threshold=0.6,  # Higher threshold to avoid false positives
                                    condition_on_previous_text=True  # Help maintain context
                                )

                                text = result["text"].strip()
                                if text:
                                    logging.info(f"Recognized: {text}")

                                    # Split and send each word
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
                                        logging.info(f"Published word: {word}")

                                    # Send full text
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
                                    logging.info(f"Published full text: {text}")

                                    # Clear buffer after successful recognition
                                    audio_buffer = []
                            except Exception as e:
                                logging.error(f"Recognition error: {e}")
                    else:
                        silence_counter += len(data) / self.sample_rate
                        if silence_counter >= min_silence_duration:
                            # Clear buffer after silence
                            audio_buffer = []
                            silence_counter = 0

                # Prevent buffer overflow
                if len(audio_buffer) > self.sample_rate * 10:  # max 10 seconds
                    audio_buffer = audio_buffer[-self.sample_rate * 5:]  # keep last 5 seconds

                time.sleep(0.01)  # Small delay to reduce CPU load

            except Exception as e:
                logging.error(f"Error in audio processing: {e}", exc_info=True)
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

    import wave

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
        """Выбор наилучшего микрофона"""
        try:
            p = pyaudio.PyAudio()
            best_device = None
            best_channels = 0
            best_rate = 0

            # Перебираем все устройства
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                if dev_info.get('maxInputChannels') > 0:  # Только устройства с входом
                    channels = dev_info.get('maxInputChannels')
                    rate = int(dev_info.get('defaultSampleRate'))

                    # Предпочитаем устройства с более высоким качеством
                    if best_device is None or (channels >= best_channels and rate >= best_rate):
                        best_device = dev_info
                        best_channels = channels
                        best_rate = rate

            if best_device:
                logging.info(f"Selected best microphone: {best_device['name']}")
                logging.info(f"Channels: {best_channels}")
                logging.info(f"Sample rate: {best_rate}")
                return int(best_device['index'])
            else:
                logging.error("No suitable microphone found")
                return None

        except Exception as e:
            logging.error(f"Error selecting microphone: {e}")
            return None

    def run(self):
        """Запуск монитора"""
        try:
            logging.info("Starting temperature and speech monitor...")

            # Инициализация детектора объектов и камеры
            self.object_detector = ObjectDetector(self.mqtt_client)
            camera_url = f"http://192.168.254.223:4747/video"  # URL для DroidCam
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
            if device_index is None:
                raise Exception("No suitable microphone found")

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
                while True:
                    try:
                        ret, frame = self.camera.read()
                        if ret:
                            processed_frame, fps, detected_objects = self.object_detector.process_frame(frame)
                            cv2.imshow("Object Detection", processed_frame)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
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
                self.print_stats()
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
    monitor = TemperatureAndSpeechMonitor()
    try:
        monitor.connect()
        monitor.run()
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise
    finally:
        monitor.cleanup()


if __name__ == "__main__":
    monitor = TemperatureAndSpeechMonitor()
    try:
        # Тестируем микрофон отдельно
        success, test_file = monitor.test_microphone(duration=5)
        if success:
            logging.info(f"Test recording saved to: {test_file}")
            input("Press Enter after checking the recording...")

        # Запускаем основной код
        monitor.connect()
        monitor.run()
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        monitor.cleanup()
