from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import time
import pytz
import torch
import logging
import uvicorn
import datetime
import schedule
from threading import Thread, Event
from api.model import Model
from lib.data_object import LoadModelRequest
from lib.base_object import BaseResponse

#############################################################################
if not os.path.exists("./audio"):
    os.mkdir("./audio")

# 配置日志记录
log_format = "%(asctime)s - %(message)s"  # 输出时间戳和消息内容
logging.basicConfig(level=logging.INFO, format=log_format)  # ['DEBUG', 'INFO']
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 配置 utc+8 時間
utc_now = datetime.datetime.now(pytz.utc)
tz = pytz.timezone("Asia/Taipei")
local_now = utc_now.astimezone(tz)

app = FastAPI(
    title="Self-checkout ASR",
    version="0.0",
    description="Self-checkout ASR",
)

model = Model()


# load the default model at startup
@app.on_event("startup")
async def load_default_model_preheat():
    """  
    Load the default model when the service starts.  
    """  
    logger.info("#####################################################")
    logger.info(f"Start to loading default model.")
    # load model
    default_language = "EN"  # EN or ZH
    model.load_model(default_language)
    logger.info(f"Default {default_language} model has been loaded successfully.")
    logger.info("#####################################################")


# load model endpoint
@app.post("/load_model")
async def load_model(request: LoadModelRequest):
    """  
    Load a specified model.  
    Available models:  
    - 'EN'  
    - 'ZH'  
      
    Parameters:  
    ----------  
    request: LoadModelRequest  
        The request object containing the model's name to be loaded  
      
    Returns:  
    ----------  
    BaseResponse  
        A response indicating the success or failure of the model loading process  
    """  
    global model

    # Load the new model
    model.load_model(request.language)
    return BaseResponse(
        message=f"{request.language} Model has been loaded successfully.", data=None
    )


# inference endpoint
@app.post("/extract_hotword")
async def transcribe(file: UploadFile = File(...)):
    """  
    Transcribe an audio file.  
      
    Parameters:  
    ----------  
    file: UploadFile  
        The audio file to be transcribed  
      
    Returns:  
    ----------  
    BaseResponse  
        A response containing the transcription results  
    """  
    # Get the file name
    file_name = file.filename
    logger.info(f"requist ID name {file_name}.")

    start = time.time()
    default_result = {"hotword": -1,}

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    audio_buffer = f"audio/{file_name}.wav"

    with open(audio_buffer, "wb") as f:
        f.write(file.file.read())

    if not os.path.exists(audio_buffer):
        logger.info("The audio file does not exist, please check the audio path.")
        return BaseResponse(message={"inference failed!"}, data=default_result)

    end = time.time()
    save_audio_time = end - start

    start = time.time()
    result, inference_time = model.transcribe(audio_buffer)
    logger.info(f"inference has been completed in {inference_time:.2f} seconds.")
    logger.info(f"| hotword: {result['hotword']} |")

    end = time.time()
    total_inference_time = end - start

    start = time.time()
    if os.path.exists(audio_buffer):
        os.remove(audio_buffer)
    end = time.time()
    remove_audio_time = end - start
    
    logger.debug(
        f"save_audio: {save_audio_time} seconds | total_inference_time: {total_inference_time} | remove_audio_time: {remove_audio_time}."
    )

    output_message = f"hotword: {result['hotword']}"
    return BaseResponse(message=output_message, data=result["hotword"])


# 清理音频文件
def delete_old_audio_files():
    """
    Delete old audio files.  
    """
    current_time = time.time()
    audio_dir = "./audio"
    for filename in os.listdir(audio_dir):
        if filename == "test.wav":
            continue
        file_path = os.path.join(audio_dir, filename)
        if os.path.isfile(file_path):
            file_creation_time = os.path.getctime(file_path)
            # 删除超过一天的文件
            if current_time - file_creation_time > 24 * 60 * 60:
                os.remove(file_path)
                logger.info(f"Deleted old file: {file_path}")


def schedule_daily_task(stop_event):
    """  
    Schedule a daily task.  
      
    Parameters:  
    ----------  
    stop_event: Event  
        The event to stop the thread  
    """  
    while not stop_event.is_set():
        if local_now.hour == 0 and local_now.minute == 0:
            delete_old_audio_files()
            time.sleep(60)  # 防止在同一分鐘內多次觸發
        time.sleep(1)


# 每日任务调度
stop_event = Event()
task_thread = Thread(target=schedule_daily_task, args=(stop_event,))
task_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    """  
    Handle shutdown event and stop the thread.  
    """  
    stop_event.set()
    task_thread.join()
    logger.info("Scheduled task has been stopped.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    uvicorn.config.LOGGING_CONFIG["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s [%(name)s] %(levelprefix)s %(message)s"
    uvicorn.config.LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s [%(name)s] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run(app, log_level="info", host="0.0.0.0", port=port)
