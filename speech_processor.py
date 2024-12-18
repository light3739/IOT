import asyncio
import json
import logging
import os
import time
import wave
from asyncio import Queue as AsyncQueue
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt
import sounddevice as sd
import whisper

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', "p4cfdde2.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv('MQTT_PORT', "8883"))
MQTT_USERNAME = os.getenv('MQTT_USERNAME', "esp8266_user")
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', "esp8266_user")
MQTT_CERT_PATH = os.getenv('MQTT_CERT_PATH', "emqxsl-ca.crt")
SPEECH_TOPIC = "speech_text"

# Audio Configuration
MICROPHONE_INDEX = 2
# В начале файла
SAMPLE_RATE = 16000  # Меняем на стандартную частоту для Whisper
CHUNK_SIZE = 1024  # Уменьшаем размер чанка
MIN_AUDIO_LENGTH = 1.5  # Увеличиваем минимальную длину
MAX_AUDIO_LENGTH = 30.0  # Увеличиваем максимальную длину
PLAYBACK_VOLUME = 0.5  # Увеличиваем громкость


# Максимальная длина аудио для распознавания


class SimpleAudioProcessor:
    def __init__(self):
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.mqtt_client.tls_set(ca_certs=MQTT_CERT_PATH)

        # Добавляем callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect

        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            logging.info(f"Connected to MQTT broker {MQTT_BROKER}")
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            raise

        # Audio setup
        self.audio_queue = AsyncQueue()
        self.stream = None
        self.output_stream = None
        self.model = whisper.load_model("small")

        # Buffer for speech recognition
        self.audio_buffer = []
        self.last_process_time = time.time()

        # Create temp directory
        self.temp_dir = "temp_audio"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def preprocess_audio(self, audio_data):
        """Предварительная обработка аудио для улучшения качества распознавания."""
        try:
            # Нормализация
            audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data

            # Удаление постоянной составляющей
            audio_data = audio_data - np.mean(audio_data)

            # Применение простого шумоподавления
            noise_threshold = 0.005
            audio_data[np.abs(audio_data) < noise_threshold] = 0

            return audio_data
        except Exception as e:
            logging.error(f"Error in audio preprocessing: {e}")
            return audio_data

    def detect_voice_activity(self, audio_data, frame_length=1024, threshold=0.01):
        """Определение наличия голоса в аудио."""
        try:
            # Разбиваем на фреймы
            frames = np.array_split(audio_data, len(audio_data) // frame_length)

            # Вычисляем энергию для каждого фрейма
            energies = [np.sum(np.abs(frame)) / len(frame) for frame in frames]

            # Определяем наличие голоса
            voice_frames = sum(1 for energy in energies if energy > threshold)
            voice_ratio = voice_frames / len(frames)

            return voice_ratio > 0.1  # Возвращаем True если более 10% фреймов содержат голос
        except Exception as e:
            logging.error(f"Error in voice detection: {e}")
            return True

    def setup_streams(self):
        try:
            # Output stream setup
            self.output_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            self.output_stream.start()

            # Input stream setup
            self.stream = sd.InputStream(
                device=MICROPHONE_INDEX,
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                callback=self.audio_callback,
                blocksize=CHUNK_SIZE
            )
            self.stream.start()

            logging.info("Audio streams initialized successfully")
        except Exception as e:
            logging.error(f"Error setting up audio streams: {e}")
            raise

    def audio_callback(self, indata, frames, time, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self.audio_queue.put(indata.copy()),
                self.loop
            )
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            client.subscribe([(SPEECH_TOPIC, 0)])
        else:
            logging.error(f"Failed to connect to MQTT broker with code: {rc}")

    def on_disconnect(self, client, userdata, rc, properties=None, reasonCode=None):
        logging.warning(f"Disconnected from MQTT broker with code: {rc}")
        if rc != 0:
            try:
                self.mqtt_client.reconnect()
                logging.info("Reconnected to MQTT broker")
            except Exception as e:
                logging.error(f"Failed to reconnect: {e}")

    def play_audio(self, audio_data):
        try:
            if self.output_stream is None:
                return

            # Apply volume
            audio_data = audio_data * PLAYBACK_VOLUME

            # Ensure correct format
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            audio_data = audio_data.astype(np.float32)

            # Play
            self.output_stream.write(audio_data)
        except Exception as e:
            logging.error(f"Error playing audio: {e}")

    async def process_speech(self, audio_data):
        try:
            # Предварительная обработка
            processed_audio = self.preprocess_audio(audio_data)

            # Проверка наличия голоса
            if not self.detect_voice_activity(processed_audio):
                logging.info("No voice activity detected, skipping processing")
                return

            temp_filename = os.path.join(
                self.temp_dir,
                f"temp_speech_{int(time.time() * 1000)}.wav"
            )

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((processed_audio * 32767).astype(np.int16).tobytes())

            # Оптимизированные параметры для Whisper
            result = await asyncio.to_thread(
                self.model.transcribe,
                temp_filename,
                language='pl',
                fp16=True,  # Включаем FP16 для ускорения
                temperature=0.0,  # Уменьшаем вариативность
                no_speech_threshold=0.5,
                compression_ratio_threshold=2.0,
                condition_on_previous_text=False  # Отключаем для ускорения
            )

            os.remove(temp_filename)

            text = result["text"].strip()
            if text:
                # Базовая постобработка текста
                text = self.postprocess_text(text)

                message = {
                    "timestamp": datetime.now().isoformat(),
                    "text": text,
                    "audio_level": float(np.abs(processed_audio).mean()),
                    "confidence": float(result.get("confidence", 0.0))
                }

                await self.send_mqtt_message(message)
            else:
                logging.info("No speech detected in audio")

        except Exception as e:
            logging.error(f"Error processing speech: {e}", exc_info=True)

    async def send_mqtt_message(self, message):
        """Асинхронная отправка MQTT сообщения с повторными попытками."""
        try:
            retry_count = 0
            max_retries = 3

            while not self.mqtt_client.is_connected() and retry_count < max_retries:
                try:
                    logging.warning("MQTT disconnected, attempting to reconnect...")
                    self.mqtt_client.reconnect()
                    await asyncio.sleep(1)
                    retry_count += 1
                except Exception as e:
                    logging.error(f"Reconnection attempt {retry_count} failed: {e}")

            if self.mqtt_client.is_connected():
                # Отправляем в топик speech_text/full
                result = self.mqtt_client.publish(
                    f"{SPEECH_TOPIC}/full",
                    json.dumps(message),
                    qos=1
                )

                # Проверяем результат отправки
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logging.info(f"Successfully sent to MQTT - Text: {message.get('text', '')}")
                    return True
                else:
                    logging.error(f"Failed to send to MQTT - Error code: {result.rc}")
                    return False
            else:
                logging.error("Failed to reconnect to MQTT broker")
                return False

        except Exception as e:
            logging.error(f"Error sending MQTT message: {e}")
            return False

    def postprocess_text(self, text):
        """Постобработка распознанного текста."""
        try:
            # Удаление лишних пробелов
            text = ' '.join(text.split())

            # Базовые правила корректировки
            text = text.replace(" ,", ",")
            text = text.replace(" .", ".")
            text = text.replace(" ?", "?")
            text = text.replace(" !", "!")

            # Приведение к правильному регистру
            if text and text[0].islower():
                text = text[0].upper() + text[1:]

            return text
        except Exception as e:
            logging.error(f"Error in text postprocessing: {e}")
            return text

    async def process_audio(self):
        while True:
            try:
                # Собираем аудио в течение 5 секунд
                start_time = time.time()
                self.audio_buffer = []

                logging.info("Starting 5-second audio collection...")

                # Используем меньший интервал записи
                while time.time() - start_time < 5.0:  # 5 секунд записи
                    try:
                        # Уменьшаем timeout для более быстрой обработки
                        data = await asyncio.wait_for(self.audio_queue.get(), timeout=0.05)
                        self.play_audio(data)
                        self.audio_buffer.extend(data.flatten())
                    except asyncio.TimeoutError:
                        continue

                # Проверяем, есть ли данные для обработки
                if len(self.audio_buffer) > SAMPLE_RATE * 0.5:  # минимум 0.5 секунды аудио
                    audio_data = np.array(self.audio_buffer)

                    # Оптимизация: предварительная фильтрация шума
                    audio_level = np.abs(audio_data).mean()
                    logging.info(f"Current audio level: {audio_level}")

                    if audio_level > 0.001:
                        # Создаем задачу для обработки речи
                        processing_task = asyncio.create_task(self.process_speech(audio_data))

                        # Ждем завершения обработки не более 10 секунд
                        try:
                            await asyncio.wait_for(processing_task, timeout=10.0)
                        except asyncio.TimeoutError:
                            logging.warning("Speech processing timed out")
                    else:
                        logging.debug(f"Audio level too low ({audio_level}), skipping...")

                # Очищаем буфер
                self.audio_buffer = []

            except Exception as e:
                logging.error(f"Error in audio processing: {e}")
                await asyncio.sleep(0.1)

    async def run(self):
        try:
            self.loop = asyncio.get_running_loop()
            self.setup_streams()

            logging.info("Starting audio processing")
            await self.process_audio()

        except Exception as e:
            logging.error(f"Error in run: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    processor = SimpleAudioProcessor()

    try:
        await processor.run()
    except KeyboardInterrupt:
        logging.info("Stopping on user request...")
    except Exception as e:
        logging.error(f"Error in main: {e}")
    finally:
        processor.cleanup()
        logging.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())
