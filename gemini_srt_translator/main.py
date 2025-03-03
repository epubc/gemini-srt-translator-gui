import typing
import json
import time
import unicodedata as ud
import srt
from srt import Subtitle
from collections import Counter
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, ContentDict
from google.generativeai import GenerativeModel

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

class GeminiSRTTranslator:
    def __init__(self, gemini_api_key: str = None, gemini_api_key2: str = None, target_language: str = None, 
                 input_file: str = None, output_file: str = None, description: str = None, 
                 model_name: str = "gemini-2.0-flash", batch_size: int = 30, free_quota: bool = True,
                 progress_callback=None):
        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.current_api_number = 1
        self.backup_api_number = 2
        self.target_language = target_language
        self.input_file = input_file
        self.output_file = output_file
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.free_quota = free_quota
        self.progress_callback = progress_callback
        self.worker = None  # Tham chiếu đến TranslationWorker

    def translate(self):
        if not self.current_api_key:
            raise Exception("Please provide a valid Gemini API key.")
        if not self.target_language:
            raise Exception("Please provide a target language.")
        if not self.input_file:
            raise Exception("Please provide a subtitle file.")
        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"

        instruction = f"You are a professional subtitle translator. Translate the following subtitles to {self.target_language}. Keep the same meaning and tone. Treat '__' as part of the text to be translated, not as a prefix to ignore. Preserve line breaks in the output. Return the translation in a JSON array with the same format as the input."
        if self.description:
            instruction += "\nAdditional user instruction: '" + self.description + "'"

        model = self._get_model(instruction)
        with open(self.input_file, "r", encoding="utf-8") as original_file, open(self.output_file, "w", encoding="utf-8") as translated_file:
            original_text = original_file.read()
            original_subtitle = list(srt.parse(original_text))
            translated_subtitle = original_subtitle.copy()
            i = 0
            total = len(original_subtitle)
            batch = []
            previous_message = None
            reverted = 0
            delay = False
            delay_time = 30

            if 'pro' in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    print("Pro model and free user quota detected, enabling 30s delay between requests...")
                else:
                    delay_time = 15
                    print("Pro model and free user quota detected, using secondary API key for additional quota...")

            lines = original_subtitle[i].content.splitlines()
            prefixed_content = "\n".join(f"__{line}" for line in lines if line.strip())
            batch.append(SubtitleObject(index=str(i), content=prefixed_content))
            i += 1
            print(f"Starting translation of {total} lines with '__' prefix...")
            if self.gemini_api_key2:
                print(f"Starting with API {self.current_api_number}:")
            if self.progress_callback:
                self.progress_callback(i, total)

            while len(batch) > 0:
                # Kiểm tra trạng thái tạm dừng
                if self.worker and self.worker.is_paused:
                    self.worker.mutex.lock()
                    while self.worker.is_paused:
                        self.worker.wait_condition.wait(self.worker.mutex)
                    self.worker.mutex.unlock()
                    
                if i < total and len(batch) < self.batch_size:
                    lines = original_subtitle[i].content.splitlines()
                    prefixed_content = "\n".join(f"__{line}" for line in lines if line.strip())
                    batch.append(SubtitleObject(index=str(i), content=prefixed_content))
                    i += 1
                    if self.progress_callback:
                        self.progress_callback(i, total)
                    continue

                try:
                    start_time = time.time()
                    previous_message = self._process_batch(model, batch, previous_message, translated_subtitle)
                    end_time = time.time()
                    print(f"Translated {i}/{total}")
                    if delay and (end_time - start_time < delay_time):
                        time.sleep(30 - (end_time - start_time))
                    if reverted > 0:
                        self.batch_size += reverted
                        reverted = 0
                        print("Increasing batch size back to {}...".format(self.batch_size))
                    if i < total and len(batch) < self.batch_size:
                        lines = original_subtitle[i].content.splitlines()
                        prefixed_content = "\n".join(f"__{line}" for line in lines if line.strip())
                        batch.append(SubtitleObject(index=str(i), content=prefixed_content))
                        i += 1
                        if self.progress_callback:
                            self.progress_callback(i, total)
                except Exception as e:
                    e_str = str(e)
                    if "quota" in e_str:
                        if self._switch_api():
                            print(f"\n🔄 API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...")
                            model = self._get_model(instruction)
                        else:
                            print("\nAll API quotas exceeded, waiting 1 minute...")
                            time.sleep(60)
                    else:
                        if self.batch_size == 1:
                            raise Exception("Translation failed, aborting...")
                        if self.batch_size > 1:
                            decrement = min(10, self.batch_size - 1)
                            reverted += decrement
                            for _ in range(decrement):
                                i -= 1
                                batch.pop()
                            self.batch_size -= decrement
                        if "Gemini" in e_str:
                            print(e_str)
                        else:
                            print("An unexpected error has occurred")
                        print("Decreasing batch size to {} and trying again...".format(self.batch_size))

            translated_file.write(srt.compose(translated_subtitle))

    def _process_batch(self, model: GenerativeModel, batch: list[SubtitleObject], previous_message: ContentDict, translated_subtitle: list[Subtitle]) -> ContentDict:
        if previous_message:
            messages = [previous_message] + [{"role": "user", "parts": json.dumps(batch)}]
        else:
            messages = [{"role": "user", "parts": json.dumps(batch)}]
        response = model.generate_content(messages)
        translated_lines: list[SubtitleObject] = json.loads(response.text)
        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")
        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            # Log nội dung gốc gửi đi
            original_content = next(x["content"] for x in batch if x["index"] == line["index"])
            print(f"Trước khi dịch (gửi đi): index={line['index']}, content='{original_content}'")
            # Log và xử lý nội dung sau khi dịch, xóa "__" ở đầu mỗi dòng
            translated_content = line["content"]
            translated_lines_cleaned = "\n".join(
                l.replace("__", "", 1) if l.startswith("__") else l 
                for l in translated_content.splitlines()
            )
            print(f"Sau khi dịch (nhận về): index={line['index']}, content='{translated_lines_cleaned}'")
            if self._dominant_strong_direction(translated_lines_cleaned) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202B{translated_lines_cleaned}\u202C"
            else:
                translated_subtitle[int(line["index"])].content = translated_lines_cleaned
        batch.clear()
        return response.candidates[0].content

    def _switch_api(self) -> bool:
        if self.current_api_number == 1 and self.gemini_api_key2:
            self.current_api_key = self.gemini_api_key2
            self.current_api_number = 2
            self.backup_api_number = 1
            return True
        if self.current_api_number == 2 and self.gemini_api_key:
            self.current_api_key = self.gemini_api_key
            self.current_api_number = 1
            self.backup_api_number = 2
            return True
        return False

    def _get_model(self, instruction: str) -> GenerativeModel:
        genai.configure(api_key=self.current_api_key)
        return genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=instruction,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

    def _dominant_strong_direction(self, s: str) -> str:
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
        ltr_count = count['L'] + count['LRE'] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"