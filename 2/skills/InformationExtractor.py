import re
import pandas as pd
from parsi_io.modules.time_extractions import TimeExtraction


class InformationExtractor:
    def __init__(self, text):
        self.text = text
        self.cleaned_text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
        self.words = self.cleaned_text.split()
        self.information = {
            "Name": None,
            "Surname": None,
            "Sex": None,
            "Age": [],
            "Job": [],
            "City": None,
            "Country": None,
            "Sickness": None,
            "Event": [],
            "Add Appointment": False,
            "New Date": None,
            "Cancel Appointment": False,
            "Cancel Date": None
        }

        # Load datasets
        self.names_df = pd.read_csv('skills//dataset//names.csv')
        self.surnames_df = pd.read_csv('skills//dataset//Surnames.csv')
        self.iranian_names_df = pd.read_csv('skills//dataset//iranianNamesDataset.csv')
        self.cities_df = pd.read_csv('skills//dataset//cities.csv')
        self.countries_capitals_df = pd.read_csv('skills//dataset//countries_capitals.csv')
        self.diseases_df = pd.read_csv('skills//dataset//diseases.csv')
        

    def extract_name(self):
        self.names_df['%GT Count of id'] = self.names_df['%GT Count of id'].str.rstrip('%').astype('float')
        max_name = None
        max_value = -float('inf')

        for word in self.words:
            if word in self.names_df['first_name'].values:
                value = self.names_df.loc[self.names_df['first_name'] == word, '%GT Count of id'].values[0]
                if value > max_value:
                    max_value = value
                    max_name = word

        if max_name:
            self.information['Name'] = max_name

    def extract_surname(self):
        surname_set = set(self.surnames_df['Surnames'].str.strip())
        for word in self.words:
            if word in surname_set:
                self.information['Surname'] = word
                break

    def extract_sex(self):
        name_to_search = self.information.get('Name')
        if name_to_search:
            row = self.iranian_names_df[self.iranian_names_df['Names'] == name_to_search]
            if not row.empty:
                gender = row['Gender'].values[0]
                self.information['Sex'] = 'مرد' if gender == 'M' else 'زن'

    def infer_category(self, tokens, clues):
        inferred = []
        for token in tokens:
            for category, keywords in clues.items():
                if any(keyword in token for keyword in keywords):
                    inferred.append(category)
        return list(set(inferred))

    def extract_job_and_age(self):
        job_clues = {
            "معلم": ["مدرسه", "دانشگاه", "کلاس", "تدریس", "آموزش", "معلم", "استاد"],
            "پزشک": ["بیمارستان", "درمانگاه", "پزشکی", "جراحی", "دکتر", "پزشک", "درمان"],
            "مهندس": ["پروژه", "مهندسی", "ساختمان", "طراحی", "مهندس", "معماری", "برنامه‌نویسی", "برق"],
            "پرستار": ["بیمار", "مراقبت", "پرستاری", "پرستار", "درمان", "بیمارستان"],
            "کارمند": ["اداره", "شرکت", "دفتر", "کارمند", "میز", "رایانه", "مدیر", "کارمندی"],
            "مدیر": ["مدیریت", "ریاست", "مدیر", "مدیریت", "سازمان", "شرکت"],
            "نویسنده": ["کتاب", "نویسندگی", "رمان", "داستان", "مجله", "نویسنده", "نوشتن", "مقاله"],
            "دانشجو": ["دانشگاه", "رشته", "تحصیل", "دانشجویی", "کلاس", "کتابخانه"],
            "وکيل": ["وکالت", "حقوق", "دادگاه", "قاضی", "قضاوت", "دادگستری", "وکیل"],
            "قاضی": ["دادگاه", "قضاوت", "قاضی", "دادگستری"],
            "خلبان": ["هواپیما", "پرواز", "خلبان", "فرودگاه"],
            "فروشنده": ["مغازه", "فروش", "فروشنده", "بازار", "فروشگاه", "مشتری"],
            "آشپز": ["رستوران", "آشپزی", "آشپز", "غذا", "پخت", "پز"],
            "هنرمند": ["هنر", "نقاشی", "هنرمند", "مجسمه‌سازی", "نمایشگاه", "گالری"],
            "بازیگر": ["فیلم", "سینما", "تئاتر", "بازیگری", "بازیگر", "نمایش"],
            "ورزشکار": ["ورزش", "تمرین", "مسابقه", "ورزشکار", "تیم", "المپیک", "باشگاه"]
        }

        age_clues = {
            "پیر": ["نوه", "بازنشستگی", "پیر", "سالخورده", "سالمند", "مسن", "کهولت", "بازنشسته"],
            "جوان": ["دانشجو", "جوان", "نوجوان", "بچه", "نوباوگان", "نوجوانی", "جوانی"],
            "میانسال": ["کار", "کارمند", "شغل", "مدیر", "میانسال", "میان‌سالی", "متوسط"]
        }

        self.information["Job"] = self.infer_category(self.words, job_clues)
        self.information["Age"] = self.infer_category(self.words, age_clues)

    def extract_city(self):
        cities_set = set(self.cities_df['name'].str.strip())
        for word in self.words:
            if word in cities_set:
                self.information['City'] = word
                break

    def extract_country(self):
        city = self.information.get("City", "")
        if city:
            row = self.countries_capitals_df[self.countries_capitals_df['Capital'] == city]
            if not row.empty:
                self.information["Country"] = row['Country'].values[0]
            else:
                self.information["Country"] = "ایران"
        else:
            self.information["Country"] = "ایران"

    def extract_sickness(self):
        diseases_set = set(self.diseases_df['Disease Name'].str.strip())
        for word in self.words:
            if word in diseases_set:
                self.information['Sickness'] = word
                break

    def extract_date_appointments(self):
        time_extractor = TimeExtraction()
        time_result = time_extractor.run(self.text)
        date_markers = time_result['markers']['datetime']

        standard_delimiters = ['.', '!', '؟', '؛', ' و ']
        pattern = '|'.join(map(re.escape, standard_delimiters))
        sentences = re.split(pattern, self.text)

        for date_range, date_text in date_markers.items():
            for sentence in sentences:
                if date_text in sentence:
                    if re.search(r'جدید', sentence):
                        self.information["Add Appointment"] = True
                        self.information["New Date"] = date_text
                    elif re.search(r'کنسل', sentence):
                        self.information["Cancel Appointment"] = True
                        self.information["Cancel Date"] = date_text

    def extract_events(self):
        event_keywords = [
            "جشنواره", "عید", "کنفرانس", "سمینار", "نمایشگاه", "مراسم", "جشن",
            "رویداد", "مسابقه", "مراسم افتتاحیه", "مراسم اختتامیه", "همایش",
            "گردهمایی", "کارگاه", "ورکشاپ", "مهمانی", "میهمانی", "شب شعر",
            "جشن سال نو", "جشن تولد", "مراسم عروسی", "مراسم عقد", "مراسم نامزدی",
            "سالگرد", "سالگرد ازدواج", "جشن خداحافظی", "مراسم یادبود", "سوگواری",
            "جشن فارغ‌التحصیلی", "مراسم تحلیف", "مراسم استقبال", "مراسم بدرقه",
            "مراسم بزرگداشت", "مراسم تجلیل", "کنسرت", "تئاتر", "پرفورمنس", "مسابقات ورزشی",
            "المپیک", "جام جهانی", "جشنواره فیلم", "جشنواره موسیقی", "جشنواره تئاتر",
            "نمایش خیابانی", "نمایش موزیکال", "نمایش کودک", "بازی کودک", "جنگ شادی",
            "مراسم شکرگزاری", "مراسم نذری", "افطار", "سحر", "حج", "عمره", "حج واجب"
        ]

        standard_delimiters = ['.', '!', '؟', '؛', ' و ']
        pattern = '|'.join(map(re.escape, standard_delimiters))
        sentences = re.split(pattern, self.text)

        for sentence in sentences:
            for keyword in event_keywords:
                if keyword in sentence:
                    self.information["Event"].append(keyword)

        self.information["Event"] = list(set(self.information["Event"]))

    def run_extractions(self):
        self.extract_name()
        self.extract_surname()
        self.extract_sex()
        self.extract_job_and_age()
        self.extract_city()
        self.extract_country()
        self.extract_sickness()
        self.extract_date_appointments()
        self.extract_events()
        return self.information


# # Example usage
# text = """ من زهرا اسدی هستم و وقتی نوە ام را به مدرسه میبردم که قبلاً در آن درس میدادم،
# کمردرد گرفتم. بعد از بازنشستگی حوصله ام سر میرود. باید از رفتن به جشنواره برج میلاد تهران صرف نظر کنم،
# چقدر حیف! در بهار برج میلاد واقعا زیبا به نظر من می رسد. برای عید نوروز برنامه داشتم ): زنگ زدم که بهتون بم
# نوبت ۱۰ اردیبهشت رو کنسل کنید و نوزدهم اردیبهشت ساعت جدید تعیین کنید. ممنونم """

# extractor = InformationExtractor(text)
# information = extractor.run_extractions()
# print(information)