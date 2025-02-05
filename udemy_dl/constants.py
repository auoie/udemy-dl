from pathlib import Path
import time

LOGIN_URL = "https://www.udemy.com/join/login-popup/?ref=&display_type=popup&loc"
LOGOUT_URL = "https://www.udemy.com/user/logout"
COURSE_URL = "https://{subdomain}.udemy.com/api-2.0/courses/{course_id}/cached-subscriber-curriculum-items?fields[asset]=results,title,external_url,time_estimation,download_urls,slide_urls,filename,asset_type,captions,media_license_token,course_is_drmed,media_sources,stream_urls,body&fields[chapter]=object_index,title,sort_order&fields[lecture]=id,title,object_index,asset,supplementary_assets,view_html&page_size=10000"
COURSE_INFO_URL = "https://{subdomain}.udemy.com/api-2.0/courses/{course_id}/"
COURSE_INFO_AND_META_URL = "https://{subdomain}.udemy.com/api-2.0/courses/{course_id}/cached-subscriber-curriculum-items?fields[asset]=results,title,external_url,time_estimation,download_urls,slide_urls,filename,asset_type,captions,media_license_token,course_is_drmed,media_sources,stream_urls,body&fields[chapter]=object_index,title,sort_order&fields[lecture]=id,title,object_index,asset,supplementary_assets,view_html&page_size=10000"
COURSE_SEARCH = "https://{subdomain}.udemy.com/api-2.0/users/me/subscribed-courses?fields[course]=id,url,title,published_title&page=1&page_size=500&search={course_name}"
SUBSCRIBED_COURSES = "https://{subdomain}.udemy.com/api-2.0/users/me/subscribed-courses/?ordering=-last_accessed&fields[course]=id,title,url&page=1&page_size=12"
MY_COURSES_URL = "https://{subdomain}.udemy.com/api-2.0/users/me/subscribed-courses?fields[course]=id,url,title,published_title&ordering=-last_accessed,-access_time&page=1&page_size=10000"
COLLECTION_URL = "https://{subdomain}.udemy.com/api-2.0/users/me/subscribed-courses-collections/?collection_has_courses=True&course_limit=20&fields[course]=last_accessed_time,title,published_title&fields[user_has_subscribed_courses_collection]=@all&page=1&page_size=1000"
QUIZ_URL = "https://{subdomain}.udemy.com/api-2.0/quizzes/{quiz_id}/assessments/?version=1&page_size=250&fields[assessment]=id,assessment_type,prompt,correct_response,section,question_plain,related_lectures"

HOME_DIR = Path().resolve()
SAVED_DIR = Path("saved").resolve()
KEY_FILE_PATH = Path("keyfile.json").resolve()
COOKIE_FILE_PATH = Path("cookies.txt").resolve()
LOG_DIR_PATH = Path("logs").resolve()
LOG_FILE_PATH = Path("logs", f"{time.strftime('%Y-%m-%d-%I-%M-%S')}.log").resolve()
LOG_FORMAT = (
    "[%(asctime)s] [%(name)s] [%(funcName)s:%(lineno)d] %(levelname)s: %(message)s"
)
LOG_DATE_FORMAT = "%I:%M:%S"
