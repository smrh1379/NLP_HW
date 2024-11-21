from opsdroid.skill import Skill
from opsdroid.matchers import match_always,match_regex
from opsdroid.events import Typing
import random
from hazm import *
import sys
import json
sys.path.insert(0, 'skills')

from InformationExtractor import *


class HelloSkill(Skill):
    # python
    example = """ من زهرا اسدی هستم و وقتی نوە ام را به مدرسه میبردم که قبلاً در آن درس میدادم،
کمردرد گرفتم. بعد از بازنشستگی حوصله ام سر میرود. باید از رفتن به جشنواره برج میلاد تهران صرف نظر کنم،
چقدر حیف! در بهار برج میلاد واقعا زیبا به نظر من می رسد. برای عید نوروز برنامه داشتم ): زنگ زدم که بهتون بم
نوبت ۱۰ اردیبهشت رو کنسل کنید و نوزدهم اردیبهشت ساعت جدید تعیین کنید. ممنونم """

    helper="سلام \n خیلی خوش اومدی \n میتونی بهم یک متن چند خطی بدی تا بتونم اطلاعات پس زمینه ای شو برات استخراج کنم \n  پیام ها باید به صورت زیر باشد: \n {""} \n".format(example)
    wrong=" پیام ارسالی کوتاه است. \n لطفا تمامی اطلاعات رو به صورت کامل در یک پیام ارسال کن \n با ارسال کلمه (راهنما) بیشتر میتونم راهنماییت کنم"
    @match_regex('.*')
    async def processing(self,message):
        text= message.text
        words=word_tokenize(text)
        
        
        if("راهنما" in words):
            await message.respond(self.helper)
        elif len(words)<15:
            await message.respond(self.wrong)
        else:
            extractor = InformationExtractor(text)
            information = extractor.run_extractions()
            for key, value in information.items():
                if isinstance(value, list) and len(value) == 1:
                    information[key] = value[0]

            await message.respond(json.dumps(information, ensure_ascii=False))

    
    # @match_regex(r'%D8%B3%D9%84%D8%A7%D9%85', case_sensitive=False, matching_condition="search")
    # async def hello(opsdroid, config, message):
    #     text = random.choice(["سلام {}", "درود {}", "سس {}"]).format(message.user)
    #     await message.respond(text)

    
    # @match_parse('hi', case_sensitive=False, matching_condition='search')
    # async def hello(self, message):
    #     text = random.choice(['Hi {}', 'Hello {}', 'Hey {}']).format(message.user)
    #     await message.respond(text)
