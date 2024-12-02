import asyncio
import logging
import time
import paho.mqtt.client as mqtt
import cv2
from video_to_text import ObjectDetector
from speech_processor import SimpleAudioProcessor

# MQTT Configuration
MQTT_BROKER = "p4cfdde2.ala.eu-central-1.emqxsl.com"
MQTT_PORT = 8883
MQTT_USERNAME = "esp8266_user"
MQTT_PASSWORD = "esp8266_user"
MQTT_CERT_PATH = "emqxsl-ca.crt"

# Глобальные переменные
running = True
mqtt_connected = False


def on_connect(client, userdata, flags, rc, properties=None):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        logging.info("Successfully connected to MQTT broker")
    else:
        logging.error(f"Failed to connect to MQTT broker with code: {rc}")


async def initialize_camera():
    """Инициализация камеры с несколькими попытками"""
    cameras_to_try = [
        ("DroidCam", "http://192.168.127.248:4747/video"),
        ("Built-in", 0),
        ("USB Camera", 1),
    ]

    for camera_name, camera_source in cameras_to_try:
        try:
            logging.info(f"Trying to initialize {camera_name} camera...")
            cap = cv2.VideoCapture(camera_source)

            if cap.isOpened():
                # Проверяем, что можем читать кадры
                ret, frame = cap.read()
                if ret and frame is not None:
                    logging.info(f"Successfully initialized {camera_name} camera")
                    # Устанавливаем параметры камеры
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
                else:
                    cap.release()
        except Exception as e:
            logging.error(f"Error initializing {camera_name} camera: {e}")

    raise Exception("Failed to initialize any camera")


async def process_camera(mqtt_client):
    """Асинхронная обработка видео"""
    global running
    cap = None
    try:
        cap = await initialize_camera()
        detector = ObjectDetector(mqtt_client)
        last_process_time = time.time()
        process_interval = 1.0  # Уменьшаем интервал для более плавной работы

        while running:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logging.error("Failed to grab frame, attempting to reinitialize camera")
                    cap.release()
                    cap = await initialize_camera()
                    continue

                current_time = time.time()

                if current_time - last_process_time >= process_interval:
                    # Обработка кадра
                    frame, fps, objects = await asyncio.get_event_loop().run_in_executor(
                        None, detector.process_frame, frame
                    )
                    if objects:
                        logging.info(f"Detected objects: {objects}")
                    last_process_time = current_time

                # Показываем кадр
                cv2.imshow('Object Detection', frame)

                # Обработка клавиш
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break
                elif key == ord('r'):
                    logging.info("Reinitializing camera...")
                    cap.release()
                    cap = await initialize_camera()

            except Exception as e:
                logging.error(f"Error in camera processing loop: {e}")
                await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"Fatal camera processing error: {e}")
        running = False
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


async def main():
    global running, mqtt_connected

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # MQTT клиент
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    mqtt_client.tls_set(ca_certs=MQTT_CERT_PATH)
    mqtt_client.on_connect = on_connect

    try:
        # Подключение к MQTT
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()

        # Ждем подключения к MQTT
        retry_count = 0
        while not mqtt_connected and retry_count < 10:
            await asyncio.sleep(1)
            retry_count += 1
            logging.info("Waiting for MQTT connection...")

        if not mqtt_connected:
            raise Exception("Failed to connect to MQTT broker")

        # Создание процессоров
        speech_processor = SimpleAudioProcessor()

        # Запуск задач
        camera_task = asyncio.create_task(process_camera(mqtt_client))
        speech_task = asyncio.create_task(speech_processor.run())

        # Ожидание завершения задач
        await asyncio.gather(camera_task, speech_task)

    except KeyboardInterrupt:
        logging.info("Stopping on user request...")
        running = False
    except Exception as e:
        logging.error(f"Error in main: {e}")
        running = False
    finally:
        running = False
        try:
            speech_processor.cleanup()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        logging.info("Cleanup completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    finally:
        cv2.destroyAllWindows()
