from glob import glob
import csv
import re
from collections import OrderedDict

'''
ETRI 전사규칙
1.2. 잡음
1.2.1. 단어의 앞과 뒤에 거의 붙어 발생된 잡음은 단어와 분리하여 표기한다.
1.2.2. 잡음이 있는 상황에서 사람에게서 발생하는 잡음은 명확히 구분될 정도로 큰 것만
표기해도 좋다.
1.2.3. 다음에 정의된 잡음 이름 뒤에 ‘/’를 붙여 표기한다.
- b : 숨소리
- l : 웃음 소리(laugh)
- o : 다른 사람의 말소리가 포함된 경우 문장의 맨 앞에 표기
- n : 주변의 잡음
'''

'''
1.1.1. 표준발성에서 벗어나거나 같은 전사에 대하여 두 가지 이상 발음이 가능한 경우 발
음전사와 철자전사를 병행하며, 이 경우 (철자전사)/(발음전사)로 표기한다 (이 문서에
서 향후 이를 '이중전사'라 칭한다).
예) (컴퓨터)/(컴퓨타) 

'''

'''
1.9.3. 문맥을 고려해봐도 전혀 알아들을 수 없는 발화는 ‘u/’ 으로 표기한다
'''

'''
interjection 간투어
'''

'''
1.5. 외국어/외래어/약자
1.5.1. 일반적으로 외국어 문자로 표기하는 경우, 통상의 발음대로 읽은 경우는 통상의 표
기를 따른다.
예) KBS, MBC, AT&T, ETRI, OPEC, FIFA 등
'''

noise_symbol_list_to_remove = ['b/', 'l/', 'n/', 'o/', 'u/']

symbol_list_to_remove = ['*', '+', '~', '♪', '”', '/']
        
interjection_dict_to_replace = {
    '아*/': '아',
    '엇/': '엇',
    '까/': '까',
    '아이/': '아이',
    '에/': '에',
    '아/': '아',
    '엇/': '엇',
    '아이/': '아이',
    '까/': '까',
    '그/': '그', 
    '어/': '어', 
    '그/': '그', 
    '아/': '아', 
    '음/': '음',
    '저/': '저',
    '저기/': '저기',
    '에/': '에',
    '으/': '으',
    '응/': '응',
    '뭐/': '뭐',
    '막/': '막',
    '오/': '오',
    '와/': '와',
    '아니/': '아니',
    '아유/': '아유',
    '야/': '야',
    '이제/': '이제',
    '아이/': '아이',
    '아예/': '아예',
    '그냥/': '그냥',
    '앙/': '앙',
    '모/': '모',
    '이/': '이',
    '한/': '한',
    '인/': '인',
    '나/': '나',
    '씨/': '씨',
    '아휴/': '아휴',
    '쪼끔/': '쪼끔',
    '하/': '하',
    '거/': '거',
    '뭐냐/': '뭐냐',
    '쫌/': '쫌',
    '팍/': '팍',
    '스/': '스',
    '뭐지/': '뭐지',
    '뭣/': '뭣',
    '엉/': '엉',
    '마/': '마',
    '뭣/': '뭣',
    '어우/': '어우',
    '인제/': '인제',
    '좀/': '좀',
    '히/': '히',
    '헤/': '헤',
    '진/': '진',
    '네/': '네',
    '마/': '마',
    '시/': '시',
    '움/': '움',
    '내/': '내',
    '이게/': '이게',
    '애/': '애',
    '근데/': '근데',
    '하유/': '하유',
    '약간/': '약간',
    '음흠/': '음흠',
    '어후/': '어후',
    '걔/': '걔',
    '응./': '응.',
    '파/': '파',
    '끄/': '끄',
    '아까/': '아까',
    '음서/': '음서',
    '유/': '유',
    '오호홍/': '오호홍',
    '요/': '요',
    '앞/': '앞',
    '살/': '살',
    '볼/': '볼',
    '그게/': '그게',
    '쭉/': '쭉',
    '쯤/': '쯤',
    '아호/': '아호',
    '광/': '광',
    '윽/': '윽',
    '알/': '알',
    '흥/': '흥',
    '쩌/.': '쩌.',
    '카/': '카',
    '후/': '후',
    '습/': '습',
    '허후/': '허후',
    '전/': '전',
    '느/': '느',
    '절/': '절',
    '휴/': '휴',
    '헛/': '헛',
    '이쿠/': '이쿠',
    '그이까/': '그이까',
    '꺄/': '꺄',
    '있/': '있',
    '오우/': '오우',
    '어호/': '어호',
    '망/': '망',
    '아우/': '아우',
    '까/': '까',
    '키/': '키',
    '자/': '자',
    '제/': '제',
    '점/': '점',
    '호/': '호',
    '착/': '착',
    '허허헝/': '허허헝',
    '에이구/': '에이구',
    '차/': '차',
    '이유이유/': '이유이유',
    '허/': '허',
    '읏/': '읏',
    '이기/': '이기',
    '화/': '화',
    '인까/': '인까',
    '홍/': '홍',
    '쪼금/': '쪼금',
    '허허/': '허허',
    '어./': '어.',
    '기/': '기',
    '악/': '악',
    '헉/': '헉',
    '형/': '형',
    '안데/': '안데',
    '이젠/': '이젠',
    '임/': '임',
    '따/': '따',
    '으흥/': '으흥',
    '수/': '수',
    '케/': '케',
    '잘/': '잘',
    '뭐시기/': '뭐시기',
    '구/': '구',
    '아이까/': '아이까',
    '걘/': '걘',
    '으휴/': '으휴',
    '어카/': '어카',
    '익/': '익',
    '쪼/': '쪼',
    '조/': '조',
    '지/': '지',
    '고/': '고',
    '여/': '여',
    '아잉/': '아잉',
    '에휴/': '에휴',
    '하이고/': '하이고',
    '아!/': '아!',
    '외/': '외',
    '급/': '급',
    '어휴/': '어휴',
    '가/': '가',
    '식/': '식',
    '아냐/': '아냐',
    '엥/': '엥',
    '아이쓰/': '아이쓰',
    '오도도도도/': '오도도도도',
    '봥/': '봥',
    '소/': '소',
    '월/': '월',
    '쇠/': '쇠',
    '즈/': '즈',
    '뭣은/': '뭣은',
    '글서/': '글서',
    '않/': '않',
    '잉/': '잉',
    '에잉/': '에잉',
    '드/': '드',
    '읍/': '읍',
    '어흐/': '어흐',
    '노/': '노',
    '삼/': '삼',
    '개/': '개',
    '열/': '열',
    '며/': '며',
    '호라/': '호라',
    '음?/': '음?',
    '아/': '아',
    '그지/': '그지',
    '압/': '압',
    '즘/': '즘',
    '근까/': '근까',
    '니/': '니',
    '아이카/': '아이카',
    '글/': '글',
    '아오./': '아오.',
    '뭔/': '뭔',
    '약/': '약',
    '미/': '미',
    '만/': '만',
    '사/': '사',
    '꽉/': '꽉',
    '워/': '워',
    '그치/': '그치',
    '옴/': '옴',
    '양/': '양',
    '아후/': '아후',
    '오호/': '오호',
    '남/': '남',
    '아이고/': '아이고',
    '아이/': '아이',
    '순/': '순',
    '또/': '또',
    '댑/': '댑',
    '얼/': '얼',
    '하또/': '하또',
    '이케/': '이케',
    '월에는/': '월에는',
    '하쒸/': '하쒸',
    '꽈/': '꽈',
    '아흐/': '아흐',
    '쾅/': '쾅',
    '어허/': '어허',
    '긋/': '긋',
    '워우/': '워우',
    '그까/': '그까',
    '억/': '억',
    '꺼휴/': '꺼휴',
    '어유/': '어유',
    '하휴/': '하휴',
    '본/': '본',
    '예/': '예',
    '크/': '크',
    '잇/': '잇',
    '아이가/': '아이가',
    '으잉/': '으잉',
    '아구/': '아구',
    '겸/': '겸',
    '뭐지?/': '뭐지?',
    '뭐야?/': '뭐야?',
    '앗/': '앗',
    '이/': '이',
    '씁/': '씁',
    '힐/': '힐',
    '상/': '상',
    '아잇/': '아잇',
    '앙까/': '앙까',
    '아후/': '아후',
    '엇/': '엇',
    '큼/': '큼',
    '백/': '백',
    '에고/': '에고',
    '흡/': '흡',
    '학년/': '학년',
    '크흠/': '크흠',
    '부/': '부',
    '허우/': '허우',
    '헐롱/': '헐롱',
    '아효/': '아효',
    '삭/': '삭',
    '별/': '별',
    '아끄/': '아끄',
    '꽤/': '꽤',
    '에./': '에.',
    '이까/': '이까',
    '아흔/': '아흔',
    '그기/': '그기',
    '와우/': '와우',
    '어떤/': '어떤',
    '쓰/': '쓰',
    '아우/': '아우',
    '우/': '우',
    '확/': '확',
    '으아/': '으아',
    '짜/': '짜',
    '음./': '음.',
    '금/': '금',
    '이쉬/': '이쉬',
    '슈슉삭/': '슈슉삭',
    '그적/': '그적',
    '오./': '오.',
    '햐/': '햐',
    '얻/': '얻',
    '장/': '장',
    '에/': '에',
    '아요/': '아요',
    '근/': '근',
    '월에도/': '월에도',
    '어어./': '어어.',
    '은/': '은',
    '영/': '영',
    '옷/': '옷',
    '우/': '우',
    '달/': '달',
    '엄/': '엄',
    '성/': '성',
    '뭔가/': '뭔가',
    '아익/': '아익',
    '헐/': '헐',
    '어이구/': '어이구',
    '회/': '회',
    '이렇게/': '이렇게',
    '으흠/': '으흠',
    '걍/': '걍',
    '언/': '언',
    '그니까/': '그니까',
    '쉬/': '쉬',
    '학/': '학',
    '아웅/': '아웅',
    '비/': '비',
    '흠/': '흠',
    '좋/': '좋',
    '아이코/': '아이코',
    '치/': '치',
    '으흠?/': '으흠?',
    '뭐냐?/': '뭐냐?',
    '머/': '머',
    '승헌쓰/': '승헌쓰',
    '왜/': '왜',
    '재/': '재',
    '연/': '연',
    '흐/': '흐',
    '끼/': '끼',
    '어머/': '어머',
    '할/': '할',
    '맨/': '맨',
    '진짜/': '진짜',
    '항/': '항',
    '술/': '술',
    '음!/': '음!',
    '아인까/': '아인까',
    '타/': '타',
    '샥/': '샥',
    '데/': '데',
    '슥/': '슥',
    '삐/': '삐',
    '무/': '무',
    '뭘/': '뭘',
    '방/': '방',
    '어즈/': '어즈',
    '아고/': '아고',
    '갸/': '갸',
    '하핫/': '하핫',
    '참/': '참',
    '을/': '을',
    '암/': '암',
    '안/': '안',
    '올/': '올',
    '웅/': '웅',
    '쫙/': '쫙',
    '캬/': '캬',
    '에햐/': '에햐',
}

word_dict_to_replace_unsorted = {
    'pc방': '피씨방',
    'PC방': '피씨방',
    'Pc방': '피씨방',
    'SNS': '에쓰 엔 에쓰',
    'tvN': '티비 엔',
    'Youtube': '유투브',
    'youtube': '유투브',
    'DMC': '디 엠 씨',
    'SBS': '에스 비 에스',
    'MT': '엠티',
    'ROTC': '알 오티씨',
    'YBM': '와이비엠',
    'k-pop': '케이팝',
    'ARS': '에이 알 에스',
    'LGU플러스': '엘지 유 플러스',
    'PT': ' 피티',
    ' KT': ' 케이티',
    ' SKT': ' 에스케이티',
    
    'sm': '에스엠',
    'jyp': '제이 와이 피',
    'JYP': '제이 와이 피',
    'CCTV': '씨씨티비',
    'LA': '엘 에이',
    'AI': '에이 아이',
    'OT': '오티',
    'JTBC': '제이티비씨',
    'ASMR': '에이 에스 엠 알',
    'CPU': '씨피유',
    'RPG': '알피지',
    'MRI': '엠 알 아이',
    'CT': '씨티',
    'KTX': '케이티엑스',
    'IFC': '아이 에프 씨',
    'x-ray': '엑스레이',
    'RG': '알지',
    'pass': '패쓰',
    'fail': '페일',
    'BJ': '비제이',
    'PPT': '피피티',
    'DC': '디씨',
    'CGV': '씨지비',
    'cgv': '씨지비',
    'sliver': '실버',
    
    'GPS': '지피에스',
    'IMF': '아이 엠 에프',
    'rpm': '알피엠',
    'CEO': '씨이오',
    'ceo': '씨이오',

    'CTO': '씨티오',
    'cto': '씨티오',

    'TMI': '티엠 아이',
    'KBS': '케이비에스',
    'cctv': '씨씨티비',
    'USB': '유에스비',
    'LED': '엘 이디',
    'led': '엘 이디',
    'PPT': '피피티',
    'ppt': '피피티',
    'CC': '씨씨',
    'DJ': '디제이',
    'rotc': '알 오 티씨',
    'DNA': '디 엔 에이',
    
    'VIP': '브이 아이 피',
    'vip': '브이 아이 피',
    'SK ': '에스 케이 ',
    'sk ': '에스 케이 ',
    'LG': '엘지',
    'lg': '엘지',
    'KT ': '케이티 ',
    'kt ': '케이티 ',
    'DSLR': '디 에스 엘 알',
    'dslr': '디 에스 엘 알',
    'CJ': '씨제이',
    'cj': '씨제이',
    'pc': '피씨',
    'VR': '브이알',
    'vr': '브이알',

    'YouTube': '유투브',
    '1101호': '천 백 일 호',
    'SUV': '에스유비',
    '오후12시': '오후 열 두 시',
    '608동': '',
    '202동': '',
    '101호': '',
    '4석': '',

    'KTX': '케이티엑스',
    'OCN': '오씨엔',
    '옛날 CG': '옛날 시지',
    'OMR': '오엠알',
    '아이폰 XR': '아이폰 엑스 알',
    '아이폰 XX': '아이폰 엑스 엑스',
    '3000대': '삼 천 대',
    'OST': '오 에스 티',
    '제 2법칙': '제 이 법칙',
    'UX': '유 엑스',
    'skt': '에스 케이 티',
    '14동': '십 사 동',
    '1201호': '천 이백 일 호',
    '111동': '백 십 일 동',
    '112호': '백 십 이 호',
    'gold': '골드',
    '주엽2동': '주엽 이동',
    '402동': '사백 이 동',
    '201호': '이백 일 호',
    '14동': '십사 동',
    '1201호': '천이백 일 호',
    '45분': '사십 오 분',
    '901-36': '구백 일 다시 삼십 육',
    '926-25번지': '구백 이십 육 다시 이십 오 번지',
    'RND': '알 엔 디',
    'TVN': '티비 엔',
    'UFC': '유 에프 씨',
    'sbs': '에스 비 에스',
    'nine': '나인',
    'CNN': '씨 엔 엔',
    'usb': '유에스비',
    'kfc': '케이에프씨',
    'homework': '홈 워크',
    'SNL': '에스 엔 엘',
    'GIF': '지 아이 에프',
    'hnm': '에이치 엔 엠',
    'cpu': '씨피유',
    'DVD': '디비디',
    'BBQ': '비비큐',
    'GTX': '지티엑스',
    'FBI': '에프비아이',
    'SF': '에스에프',
    'UFO': '유에프오',
    'asmr': '에이에스 엠 알',
    'BTS': '비티에스',
    'OEM': '오이엠',
    'RPG': '알피지',
    'rpg': '알피지',
    'BHC': '비에이치씨',
    'SRT': '에스알티',
    'srt': '에스알티',
    'one': '원',
    'LP': '엘피',

    'IS': '아이에스',
    'eu': '이 유',
    'EU': '이 유',
    '545일': '오백 사십 오 일',
    '1839년': '천 팔백 삼십 구 년',
    '430명': '사백 삼십 명',
    '56백': '오륙백',
    '13000km': '만 삼 천 킬로미터',
    '3000세대': '삼천 세대',
    'JUST DO IT': '저스트 두 잇',
    ',5만원권': ', 오만원권',
    'kt': '케이티',
    'sk': '에스케이',
    '%': '퍼센트',
    '26명': '스물 여섯 명',
    '바비큐치킨이1개': '바비큐치킨 한 개',

    '107-601호': '백칠 다시 육 백 일 호',
    '22길': '이십 이 길',
    '20-28': '이십 다시 이십팔',
    'suv차량': '에스 유 비 차량',
    '15%할인': '십 오 프로 할인',

    '오후1시': '오후 한 시',
    '오후2시': '오후 두 시',
    '오후3시': '오후 세 시',
    '오후4시': '오후 네 시',
    '오후5시': '오후 다섯 시',
    '오후6시': '오후 여섯 시',
    '오후7시': '오후 일곱 시',
    '오후8시': '오후 여덟 시',
    '오후9시': '오후 아홉 시',
    '오후10시': '오후 열 시',
    '오후11시': '오후 열 한시',

    '저녁6시': '저녁 여섯 시',
    '저녁7시': '저녁 일곱 시',
    '저녁8시': '저녁 여덟 시',
    '저녁9시': '저녁 아홉 시',
    '저녁10시': '저녁 열 시',
    '저녁11시': '저녁 열 한 시',
        
    '아침7시': '아침 일곱 시',
    '아침8시': '아침 여덟 시',
    '아침9시': '아침 아홉 시',
    '아침10시': '아침 열 시',
    
    ' 1시': ' 한 시',
    ' 2시': ' 두 시',
    ' 3시': ' 세 시',
    ' 4시': ' 네 시',
    ' 5시': ' 다섯 시',
    ' 6시': ' 여섯 시',
    ' 7시': ' 일곱 시',
    ' 8시': ' 여덟 시',
    ' 9시': ' 아홉 시',
    '10시': '열 시',
    '11시': '열 한 시',
    # '12시': '열 두 시',

    ' 1명': ' 한 명',
    ' 2명': ' 두 명',
    ' 3명': ' 세 명',
    ' 4명': ' 네 명',
    ' 5명': ' 다섯 명',
    ' 6명': ' 여섯 명',
    ' 7명': ' 일곱 명',
    ' 8명': ' 여덟 명',
    ' 9명': ' 아홉 명',
    ' 10명': ' 열 명',

    ' 1월': ' 일 월',
    ' 2월': ' 이 월',
    '3월': '삼 월',
    '4월': '사 월',
    '5월': '오 월',
    '6월': '유 월',
    '7월': '칠 월',
    '8월': '팔 월',
    '9월': '구 월',
    '10월': '시 월',
    '11월': '십일 월',
    '12월': '십이 월',

    '월1일': '월 일일',
    '월2일': '월 이일',
    '월3일': '월 삼일',
    '월4일': '월 사일',
    '월5일': '월 오일',
    '월6일': '월 육일',
    '월7일': '월 칠일',
    '월8일': '월 팔일',
    '월9일': '월 구일',
    '월10일': '월 십일',
    '월11일': '월 십일일',
    '월12일': '월 십이일',
    '월13일': '월 십삼일',
    '월14일': '월 십사일',
    '월15일': '월 십오일',
    '월16일': '월 십육일',
    '월17일': '월 십칠일',
    '월18일': '월 십팔일',
    '월19일': '월 십구일',
    '월20일': '월 이십일',
    '월21일': '월 이십일일',
    '월22일': '월 이십이일',
    '월23일': '월 이십삼일',
    '월24일': '월 이십사일',
    '월25일': '월 이십오일',
    '월26일': '월 이십육일',
    '월27일': '월 이십칠일',
    '월28일': '월 이십팔일',
    '월29일': '월 이십구일',
    '월30일': '월 삼십일',
    '월31일': '월 삼십일일',

    '33개국': '삼십 삼 개국',
    '어른10': '어른 열',
    '성인3명': '성인 세 명',
    '차량 3대': '차량 세 대',
    'A세트': '에이 세트',
    'a세트': '에이 세트',
    'BC카드': '비씨 카드',
    'gs': '지에스',
    'TN': '티 엔',
    'gta': '쥐 티 에이',
    'TV': '티비',
    'a, b, c, d': '에이, 비, 씨, 디',
    's세븐': '에스 세븐',
    'JY': '제이 와이',
    'VS': '브이 에스',
    'PCM': '피씨엠',
    'imf': '에이 엠 에프',
    'IMF': '에이 엠 에프',
    'GD': '쥐 디',
    'EXID': '이 엑스 아이 디',
    '23도': '이십 삼 도',
    '1999년': '천 구백 구십 구 년',
    '33일째': '삼십 삼 일 째',
    '90킬로': '구십 킬로',
    '450대': '사백 오십 대',
    '1901년': '천 구백 일 년',
    '1996년': '천 구백 구십 육 년',
    'G20': '쥐 이 십',
    '200채': '이 백 채',
    'UST': '유 에스 티',
    '360바퀴': '삼백 육십 바퀴',
    ' 83년': ' 팔십 삼 년',
    ' 89일': ' 팔십 구 일',
    '전국 17개': '전국 열 일곱 개',
    '제1호': '제 일 호',
    '34가지': '서른 네 가지',
    '250톤': '이백 오십 톤',
    '625 전쟁': '육 이 오 전쟁',
    '160여년': '백 육십 여 년',
    'n명': '엔 명',
    'n시': '엔 시',
    '298-163': '이백 구십 팔 다시 백 육십 삼',
    'suv': '에스 유 비',
    '아이1명': '아이 한 명',
    '어른2명': '어른 두 명',
    '101-1': '백 일 다시 일',
    'u플러스': '유 플러스',

    't멤버십': '티 멤버십',
    't멤버쉽': '티 멤버쉽',
    't 멤버십': '티 멤버십',
    't 멤버쉽': '티 멤버쉽',

    'kt멤버십': '케이티 멤버십',
    'kt멤버쉽': '케이티 멤버쉽',
    'kt 멤버십': '케이티 멤버십',
    'kt 멤버쉽': '케이티 멤버쉽',
    'skt멤버십': '에스케이티 멤버십',
    'skt멤버쉽': '에스케이티 멤버쉽',
    'skt 멤버십': '에스케이티 멤버십',
    'skt 멤버쉽': '에스케이티 멤버쉽',

    '석수3동': '석수 삼 동',
    '123번지': '백 이십 삼 번지',
    ' 603호': ' 육백 삼 호',
    ' 17명': ' 열 일곱 명',
    '파스타1개': '파스타 한 개',
    '123-12': '백 이십 삼 다시 십 이',
    '12시-1시': '열 두 시에서 한 시',
    '298-163번지': '이백구십팔 다시 백 육십 삼 번지',
    '여자2명': '여자 두 명',
    '15퍼센트': '십 오 퍼센트',
    '2째주': '둘째 주',
    # '010-4564-1212': '',
    # '010-2570-2570': '',
    # '010-7878-8282': '',
    # '010-6890-1111': '',
    '음..6시': '음. 여섯 시',
    'DS': '디 에스',
    'C샾': '씨 샾',
    'C형': '씨 형',
    'C언어': '씨 언어',
    'C 언어': '씨 언어',
    'B 언어': '비 언어',
    '몇 kg': '몇 킬로',
    'hb': '에이치 비',

    'UN': '유엔',
    'un': '유엔',

    '차량3대': '차량 세 대',
    'ACSM': '에이 씨 에스 엠',
    'ASM': '에이 에스 엠',
    'ipk': '아이 피 케이',
    'N년': '엔 년',
    'a랜드': '에이 랜드',
    '27페이지': '스물 일곱 페이지',
    '제1회': '제 일 회',
    'OECD': '오이씨디',
    'GDP': '쥐디피',

    '1978년': '천 구백 칠십 팔 년',
    '1986년': '천 구백 팔십 육 년',
    '1914년': '천 구백 십 사 년',
    '1995년': '천 구백 구십 오년',
    '1919년': '천 구백 십 구 년',
    'A라는': '에이라는',
    '36곳': '서른 여섯 곳',
    'TED': '테드',
    '성인4명': '성인 네 명',
    '아이들2명': '아이들 두 명',
    '빌딩2층': '빌딩 이 층',
    '0월 0일 0요일 0시': '땡 월 땡 일 땡 요일 땡 시',
    '유치원생1명': '유치원생 한 명',
    '15~20분': '십오 분에서 이십 분',
    'VOD': '비오디',
    'vod': '비오디',
    'A 마이너': '에이 마이너',
    'akb': '에이 케이 비',
    'as': '에이 에스',
    'AS': '에이 에스',
    'YMCA': '와이 엠 씨 에이',
    'GSAT': '쥐사트',
    'B급': '비 끕',
    'LS': '엘 에스',
    'she is teenager': '쉬 이즈 티네이져',
    'BMI': '비엠아이',
    'bmi': '비엠아이',
    'MLB': '엠 엘 비',
    'mlb': '엠 엘 비',
    'OST': '오에스티',
    'ost': '오에스티',
    'KFC': '케이에프씨',
    'kfc': '케이에프씨',
    'IBK': '아이비케이',
    'ibk': '아이비케이',

    '비타민 B': '비타민 비',
    '비타민 C': '비타민 씨',
    '비타민 D': '비타민 디',

    '비타민 b': '비타민 비',
    '비타민 c': '비타민 씨',
    '비타민 d': '비타민 디',

    'Sold Out': '솔드 아웃',
    'CPR': '씨피알',
    'PMP': '피엠피',
    'FPS': '에프피에스',
    'fps': '에프피에스',
    'IP': '아이피',
    'Interesting': '인터레스팅',
    'dm': '디엠',
    'DM': '디엠',
    'TT': '티 티',
    'Grand Canyon': '그랜드 캐니언',
    'A, B, C조': '에이, 비, 씨 조',
    'a,b,c,d': '에이, 비, 씨, 디',
    'A, B, C, D': '에이, 비, 씨, 디',
    'A형 독감': '에이형 독감',
    'TGI': '티 쥐 아이',
    'IT': '아이 티',
    'XS': '엑스 에스',
    'PDF': '피디에프',
    'pdf': '피디에프',
    'xr': '엑스 알',
    'xs': '엑스 에스',
    'ADHD': '에이 디 에이치 디',
    'REC': '알 이 씨',
    'a플': '에이 플',
    'b플': '비 플',
    'MD': '엠디',
    'MMO': '엠 엠 오',
    'LTE': '엘 티 이',
    'CU': '씨유',
    'cu': '씨유',
    'PPR': '피피 알',
    'CBS': '씨 비 에스',
    'HD': '에이치 디',
    'Nike': '나이키',
    'EDM': '이 디 엠',
    'edm': '이 디 엠',
    'MBA': '엠 비 에이',
    'PX': '피 엑스',
    'MMR': '엠 엠 알',
    'mmr': '엠 엠 알',
    'HSK': '에이치 에스 케이',
    'TWO': '투',
    'ONE': '원',
    'VJ 특공대': '브이 제이 특공대',
    'MCM': '엠 씨 엠',
    'KDPP': '케이 디 피피',
    'DDP': '디디피',

    'ATM': '에이 티 엠',
    'atm': '에이 티 엠',
    'ABC마트': '에이 비 씨 마트',
    '태권V': '태권 브이',

    'VPN': '브이 피 엔',
    'vpn': '브이 피 엔',
    'OMG': '오 엠 쥐',
    'rnb': '알 엔 비',
    'bhc': '브이 에이치 씨',
    'sns': '에스 엔 에스',
    'sightseeing': '사이트시잉',
    'YG': '와이 쥐',
    'NC백화점': '엔씨 백화점',
    'IMAX': '아이맥스',
    'X-ray': '엑스 레이',
    'uplus': '유 플러스',
    'B, C, D': '비, 씨, 디',
    'KC대학교': '케이씨 대학교',

    'IT': '아이티',
    'it': '아이티',

    'FTA': '에프 티 에이',
    'fta': '에프 티 에이',
    
    'SKY': '스카이',
    'sky': '스카이',

    'DNA': '디 엔 에이',
    'dna': '디 엔 에이',

    'OK': '오케이',
    'ok': '오케이',
    
    'A형': '에이 형',
    'B형': '비 형',
    'AB형': '에이비 형',
    'O형': '오 형',

    'a형': '에이 형',
    'b형': '비 형',
    'ab형': '에이비 형',
    'o형': '오 형',
    
    'nba': '엔 비 에이',
    'NBA': '엔 비 에이',

    'TO': '티 오',

    'MIT': '엠 아이 티',
    'ITX': '아이 티 엑스',
    'SOS': '에스 오 에스',
    'PC 게임': '피씨 게임',

    '90 이상': '구십 이상',
    '10나노미터': '십 나노미터',
    '10가구': '열 가구',
    '1930년': '천 구백 삼십 년',
    '1847년': '천 팔백 사십 칠 년',
    '나머지 60': '나머지 육십',

    '백 분의 25': '백 분의 이십오',
    'gtl': '쥐티엘',
    'cs': '씨에스',
    'CS': '씨에스',
    '3보이상': '삼 보 이상',
    '80이 넘으신': '여든이 넘으신',
    '38년': '삼십 팔 년',
    '34개': '서른 네 개',
    ' 230년': ' 이 삼십년',
    'CNC': '씨 엔 씨',
    'cnc': '씨 엔 씨',
    'MC': '엠씨',
    'gtq': '쥐 티 큐',
    'GTQ': '쥐 티 큐',
    'DREAM': '드림',
    'MBC': '엠비씨',
    'mbc': '엠비씨',
    'SD카드': '에스 디 카드',
    'sd카드': '에스 디 카드',
    'SM': '에스 엠',
    'sm': '에스 엠',
    'A조, B조, C조, D조': '',
    '세트 A': '세트 에이',
    '몇 m': '몇 미터',
    'MSG': '엠 에스 쥐',
    'msg': '엠 에스 쥐',
    'rt': '알티',
    'RT': '알티',
    'RO': '알 오',
    'ROH': '알 오 에이치',
    'GM': '쥐엠',
    'gm': '쥐엠',
    'AK': '에이 케이',
    'ak': '에이 케이',
    'HP': '에이치 피',
    'hp': '에이치 피',
    'ot': '오티',
    'OT': '오티',
    'hn': '에이치 엔',
    'HN': '에이치 엔',
    'GS': '쥐에스',
    'gs': '쥐에스',
    'MS': '엠 에스',
    'ms': '엠 에스',
    'CTS': '씨 티 에스',
    'cts': '씨 티 에스',
    'ANB': '에이 엔 비',
    'anb': '에이 엔 비',
    'AOS': '에이 오 에스',
    'aos': '에이 오 에스',
    '게임 CD': '게임 씨디',
    '게임 cd': '게임 씨디',
    'NCA': '엔 씨 에이',
    'IBF': '아이 비 에프',
    'IQ': '아이 큐',
    'knk': '케이 엔 케이',
    'MAX': '맥스',
    'BEM': '비 이 엠',
    'LPG': '엘 피 쥐',
    'lpg': '엘 피 쥐',
    'LC': '엘 씨',
    'lc': '엘 씨',
    'RC': '알 씨',
    'rc': '알 씨',
    'MRA': '엠 알 에미',
    'kb': '케이 비',
    'DIY': '디 아이 와이',
    'diy': '디 아이 와이',
    'npc': '엔 피 씨',
    'NPC': '엔 피 씨',
    'ip': '아이피',
    'IP': '아이피',
    'ti': '티 아이',
    'omr': '오 엠 알',
    'OMR': '오 엠 알',
    'mp': '엠피',
    'pt': '피티',
    'PT': '피티',
    'nine to six': '나인 투 식스',
    'PC게임': '피씨 게임',
    'pc게임': '피씨 게임',
    'API': '에이 피 아이',
    'api': '에이 피 아이',
    'ET': '이티',
    'et': '이티',
    'MC': '엠씨',
    'mc': '엠씨',
    'VDL': '브이 디 엘',
    'vdl': '브이 디 엘',
    'CS': '씨 에스',
    'cs': '씨 에스',
    'CSI': '씨 에스 아이',
    'csi': '씨 에스 아이',
    'ktx': '케이 티 엑스',
    'KTX': '케이 티 엑스',
    'JLPT': '제이 엘 피 티',
    'jlpt': '제이 엘 피 티',
    'JPT': '제이 피 티',
    'jpt': '제이 피 티',
    'SE': '에스 이',
    'FC서울': '에프 씨 서울',
    'SDS': '에스 디 에스',
    'sds': '에스 디 에스',
    'SK': '에스 케이',
    'sk': '에스 케이',
    'sf': '에스 에프',
    'SF': '에스 에프',
    'bj': '비제이',
    'BJ': '비제이',
    'bts': '비티에스',
    'BTS': '비티에스',
    'MG': '엠 쥐',
    'mg': '엠 쥐',
    'PG': '피 쥐',
    'pg': '피 쥐',
    'how to made idea to money?': '하우 투 메이드 아이디어 투 머니?',
    'CF': '씨 에프',
    'cf': '씨 에프',
    'CG': '씨 쥐',
    'cg': '씨 쥐',
    'GS': '쥐 에스',
    'gs': '쥐 에스',
    'IOT': '아이오티',
    'iot': '아이오티',
    'PPL': '피 피 엘',
    'ppl': '피 피 엘',
    'FM': '에프 엠',
    'fm': '에프 엠',
    'Would like something to, something to drink?': '우쥬 라이크 썸띵 투, 썸띵 투 드링크?',
    'identity': '아이덴티티',
    'KGB': '케이 쥐 비',
    'kgb': '케이 쥐 비',
    'ncs': '엔 씨 에스',
    'NCS': '엔 씨 에스',
    'IOS': '아이 오 에스',
    'ios': '아이 오 에스',
    'PSAT': '피샛',
    'psat': '피샛',
    'vlog': '브이 로그',
    'Once a gone never back again': '원스 어 건 네버 백 어게인',
    'EPL': '이피엘',
    'epl': '이피엘',
    'ten': '텐',
    'rain': '레인',
    'refresh': '리프레쉬',
    'screen X': '스크린 엑스',
    'Screen X': '스크린 엑슷',
    'why': '와이',
    'Otago': '오타고',
    'station': '스테이션',
    'QR': '큐알',
    'qr': '큐알',
    'px': '피엑스',
    'PX': '피엑스',
    '토익 850': '토익 팔 백 오십',
    'sunset': '썬 쎗',
    'not bad': '낫 뱃',
    'Clock Tower': '클락 타워',
    'Live': '라이브',
    'tvn': '티비엔',
    'TVN': '티비엔',
    'MBTI': '엠비티아이',
    'mbti': '엠비티아이',
    'self printing': '셀프 프린팅',
    'Dunedin': '더니든',
    'Auckland': '오클랜드',
    'Victoria': '빅토리아',
    'university': '유니버시티',
    'University': '유니버시티',
    'g마켓': '지마켓',
    'G마켓': '지마켓',
    'Daily Moisture Therapy': '데일리 모이스쳐 테라피',
    'HANDLE': '핸들',
    'Octagon': '옥타곤',
    'A, B, C': '에이, 비, 씨',
    'go away': '고 어웨이',
    'liquid': '리퀴드',
    'simple': '심플',
    'Party People': '파티 피플',
    'fox': '폭스',
    'function': '펑션',
    'Sorry': '쏘리',
    'sorry': '쏘리',
    'Do you wanna fight?': '두 유 워너 파이트?',
    'X맨': '엑스맨',
    'x맨': '엑스맨',
    'subway': '서브웨이',
    'fun': '펀',
    'New York': '뉴욕',
    'ranking': '랭킹',
    'Window': '윈도우',
    'MAC': '맥',
    'airbnb': '에어비엔비',
    'ping-pong': '핑 퐁',
    'personal training': '펄스널 트레이닝',
    'orientation training': '오리엔테이션 트레이닝',
    'Sentence': '센텐스',
    'for me': '포 미',
    'no': '노',
    'way back home': '웨이 백 홈',
    'what are you doing?': '왓 알 유 두잉?',
    'Serious': '시리어스',
    'yes or yes': '예스 오어 예스',
    'apk': '에이 피 케이',
    'APK': '에이 피 케이',
    'ddp': '디디피',
    'program': '프로그램',
    'Program': '프로그램',
    'Do you wanna fight?': '두 유 워너 파이트?',
    'insurance': '인슈런스',
    'three': '쓰리',
    'six': '씩스',
    'why': '와이',
    'dslr': '디 에스 엘 알',
    'DSLR': '디 에스 엘 알',
    'OPIC': '오픽',
    'opic': '오픽',
    'eight': '에잇',
    'rnd': '알 엔 디',
    'Basket Ball for Life': '배스킷 볼 포 라이프',
    'little bear': '리틀 베어',
    'FIFA': '피파',
    'fifa': '피파',
    'Call me by your name': '콜 미 바이 욜 네임',
    'Understand': '언더스탠드',
    'break down': '브레이크 당운',
    'mc the max': '엠씨 더 맥스',
    'a/s': '에이에스',
    'A/S': '에이에스',
    'as': '에이에스',
    'AS': '에이에스',
    'JEEP': '짚',
    'Game of Thrones': '게임 오브 쓰론',
    'English': '잉글리쉬',
    'english': '잉글리쉬',
    'Are you OK?': '알 유 오케이?',
    'Are you ok?': '알 유 오케이?',
    'lake': '레이크',
    'nine to five': '나인 투 파이브',
    'p2p': '피 투 피',
    'P2P': '피 투 피',
    'studio': '스튜디오',
    'she': '쉬',
    'Adidas': '아디다스',
    'adidas': '아디다스',
    'you need to talk': '유 니드 투 턱',
    'SUM': '썸',
    'AVERAGE': '에버리지',
    'talk': '턱',
    'MINI': '미니',
    'hierarchy': '하이에라키',
    'sat': '에스에이티',
    'SAT': '에스에이티',

    'kind': '카인드',
    'yolo': '욜로',

    '5~6명 정도': '다여섯명 정도',
    '6~7명에': '여서 일곱 명에',
    '통신사 / 멤버십': '통신사 멤버십',
    '하나요>': '하나요?',
    '시30분': '시 30분',
    '자리2개': '자리 두 개',
    'bow': '바우',
    'tv': '티비',
    'TV': '티비',
    'dry food': '드라이 푸드',
    'lol': '엘 오 엘',
    'VJ특공대': '브이 제이 특공대',
    'vj특공대': '브이 제이 특공대',
    'Girl I swear': '걸 아이 스웨어',
    'yes or no question': '예스 오어 노 퀘스쳔',
    'kb': '케이비',
    'KB': '케이비',
    'sure': '숼',
    'Sure': '숼',
    'Why not': '와이 낫',
    'practical': '프랙티컬',
    'GTA': '쥐 티 에이',
    'gta': '쥐 티 에이',
    'Your name is Dumbledore': '유월 네임 이즈 덤블도어',
    'Party people': '파티 피플',
    'success': '썩쎄스',
    'ai': '에이 아이',
    'AI': '에이 아이',
    'SPSS': '에스 피 에스에스',
    'hsk': '에이취 에스 케이',
    'HSK': '에이취 에스 케이',
    'shape of you': '쉐이프 오브 유',
    'EBS': '이 비 에스',
    'ebs': '이 비 에스',
    'IC': '아이 씨',
    'ic': '아이씨',
    'all of my life': '올 오브 마이 라이프',
    'PSY': '싸이',
    'know': '노',
    'stereo type': '스테레오 타입',
    '헌터 x 헌터': '헌터 바이 헌터',
    'Happy dance day to you': '해피 댄스 데이 투 유',
    'extended cut': '익스텐디드 컷',
    '12 월': '십 이 월',
    'FC': '에프 씨',
    'fc': '에프 씨',
    'I WANT YOU BACK': '아이 원 유 백',
    'LTE': '엘 티 이',
    'lte': '엘 티 이',
    'N수': '엔 수',
    'n수': '엔 수',
    '그러sl까': '그러니까',
    'bubble gum': '버블 검',
    'too much information': '투 머치 인포메이션',
    'is not': '이즈 낫',
    'not': '낫',
    'out of the blue like': '아웃 오브 더 블루 라이크',
    'mother tongue': '마더 텅',
    'EXO': '엑소',
    'racist': '레이시스트',
    'be': '비',
    'Shape On My Heart': '쉐이프 온 마이 하트',
    'shape of you': '쉐이프 오브 유',
    'mistake': '미스테이크',
    'everybody': '에브리바디',
    'la': '엘 에이',
    'LA': '엘 에이',
    'introduction': '인트로덕션',
    'FILA': '필라',
    'fila': '필라',
    'adverti': '어드벌티',
    'remind': '리마인드',
    'five': '파이브',
    'supprtive': '서포티브',
    'oecd': '오이씨디',
    'BASE': '베이스',
    'LEARNING': '러닝',
    'JBJ': '제이비제이',
    'jbj': '제이비제이',
    'PD': '피디',
    'pd': '피디',
    'mt': '엠티',
    'MT': '엠티',
    'HALSEY': '에이치 에이 엘 에스 이 와이',
    'CBT': '씨티비',
    'cbt': '씨비티',
    'e xcept for': '익셉트 포',
    'BMW': '비엠더블유',
    'bmw': '비엠더블유',
    'KBG': '케이비쥐',
    'kbg': '케이비쥐',
    'D.C.': '디.씨.',
    'jtbc': '제이티비씨',
    'JTBC': '제이티비씨',
    'Text Me': '텍스트 미',
    'DHC': '디 에이치 씨',
    'dhc': '디 에이치 씨',
    'pin point': '핀 포인트',
    'SSD': '에스에스디',
    'ssd': '에스에스디',
    'ucc': '유씨씨',
    'UCC': '유씨씨',
    'YB': '와이비',
    'yb': '와이비',
    'OB': '오비',
    'ob': '오비',
    'js': '제이 에스',
    'JS': '제이 에스',
    'NC소프트': '엔씨 소프트',
    'nc소프트': '엔씨 소프트',
    'MBN': '엠비엔',
    'mbn': '엠비엔',
    'SDI': '에스 디 아이',
    'sdi': '에스 디 아이',
    'chain': '체인',
    'vat': '브이 에이 티',
    'ICT': '아이 씨 티',
    'ict': '아이 씨 티',
    'enjoyable': '인조이어블',
    'JOLLY': '졸리',
    'yes or no': '예스 오어 노',
    'BBC': '비비씨',
    'bbc': '비비씨',
    'JSA': '제이 에스 에이',
    'jsa': '제이 에스 에이',
    'restaurant': '레스토랑',
    'sold out': '솔드 아웃',
    'N빵': '엔 빵',
    'n빵': '엔 빵',
    'ID': '아이디',
    'id': '아이디',
    'LTE': '엘티이',
    'lte': '엘티이',
    'GRE': '쥐 알 이',
    'gre': '쥐 알 이',
    'octago': '옥타고',
    'BAD BOY': '배드 보이',
    'Problem Base Learning': '프라블렘 베이스 러닝',
    'k pop': '케이 팝',
    'X RAY': '엑스 레이',
    'good': '굿',
    'self respect': '셀프 리스펙트',
    'YES OR YES': '예스 오어 예스',
    'WHO': '더블유 에이치 오',
    'assignment': '어싸인먼트',
    'fixar': '픽사',
    'ZARA': '자라',
    'SPAO': '스파오',
    'XR': '엑스 알',
    'xr': '엑스 알',
    'Gucci': '구찌',
    'VLOOK': '브이 룩',
    'HLOOK': '에이치 룩',
    'dj': '디제이',
    'DJ': '디제이',
    'amd': '에이 엠 디',
    'AMD': '에이 엠 디',
    'UCLA': '유 씨 엘 에이',
    'ucla': '유 씨 엘 에이',
    'DID': '디 아이 디',

    '100분의 25': '백 분의 이십 오',
    '97의 에너지': '구십 칠의 에너지',
    '여러분들안에 97가 있어요': '여러분들안에 구십 칠이 있어요',
    '엠피3': '엠피 쓰리',
    ' 85가': '팔십 오가',
    '3만3천개': '삼만 삼천 개',
    '이걸 계속 12년 23 년 더 하기위해서': '이걸 계속 일 이 년 이 삼 년 더 하기 위해서',

    'express': '익스프레스',
    'WHAT SHOULD I DO': '왓 슈드 아이 두',
    'SHOULD I': '슈드 아이',
    'We are': '위 아',
    'dry': '드라이',
    'LOVE': '러브',
    'GDP': '쥐 디 피',
    'gdp': '쥐 디 피',
    'Xr': '엑스 알',
    'MR': '엠 알',
    'cpa': '씨 피 에이',
    'CPA': '씨 피 에이',
    'SR': '에스 알',
    'AR': '에이 알',
    'PR': '피 알',
    'YES24': '예스 이십 사',
    'HKS': '에이치 케이 에스',
    'CA': '씨 에이',
    'Thank you': '땡 큐',
    'XY': '엑스 와이',
    'MX': '엠 엑스',
    'mx': '엠 엑스',
    'GEN.G': '젠지',
    'tmi': '티 엠 아이',
    'TMI': '티 엠 아이',
    'We Will Rock You': '위 윌 락 유',
    'yg': '와이 지',
    'YG': '와이 지',
    'abc마트': '에이 비 씨 마트',
    'ABC마트': '에이 비 씨 마트',
    'K-POP': '케이 팝',
    '그q럼': '그럼',
    'SW': '소프트웨어',
    'sw': '소프트웨어',
    '1녀': '일녀',
    'Xerostomia': '제로스토미아',
    '3항': '삼 항',
    'g마켄': '지 마켄',
    '그r거': '그거',
    'nc백화점': '엔씨 백화점',
    'NC백화점': '엔씨 백화점',
    'ocn': '오 씨 엔',
    'OCN': '오 씨 엔',
    'IBM': '아이 비 엠',
    'ibm': '이아 비 엠',
    'ng': '엔 지',
    'NG': '엔 지',
    'tmi': '티 엠 아이',
    'TMI': '티 엠 아이',
    'DC': '디씨',
    'dc': '디씨',
    'OS': '오 에스',
    'os': '오 에스',
    '120이라고 치면': '백 이십이라고 치면',
    '괜찮eo': '괜찮대',
    '영민이eh': '영민이도',
    '3.5': '삼 쩜 오',
    'One': '원',
    '거dp요': '거에요',
    'IF': '이프',
    'if': '이프',
    '(방문 포장, 남은 음식 포장 등)': '방문 포장, 남은 음식 포장 등',
    '(분실물)': '분실물',
    '(특정 상황)': '특정 상황',
    '010-4564-1212': '공일공 사오육사 일이일이',
    '010-2828-8282': '공일공 이팔이팔 팔이팔이',
    '010-2570-2570': '공일공 이오칠팔 이오칠공',
    '010-6890-1111': '공일공 육팔구공 일일일일',
    '010-4560-1230': '공일공 사오육공 일이삼공',
    '010-7878-8282': '공일공 칠팔칠팔 팔이팔이',
    'Take a trip': '테이크 어 트립',
    'PC': '피씨',
    'pc': '피씨',
    'GP': '지피',
    'gp': '지피',
    'GOP': '지오피',
    'gop': '지오피',
    'Esc': '이 에스 씨',
    'SHOW ME THE MONEY': '쇼 미 더 머니',
    'PRODUCE': '프로듀스',
    'GO': '고',
    'CIA': '씨아이에이',
    'armory': '아모리',

    '한 30명': '한 삼십 명',
    ' 14라고 적어': ' 십사라고 적어',

    'e-class': '이 클래스',
    '이-클래스': '이 클래스',

    '2002': '이천이',

    '지금 20': '지금 스물',

    ' 11만': ' 십일만',

    '엑스-레이': '엑스레이',
    'X-레이': '엑스레이',

    '케이-팝': '케이팝',
    'K-POP': '케이팝',

}

word_dict_to_replace = OrderedDict()

for sorted_key in reversed(sorted(word_dict_to_replace_unsorted, key = lambda x: len(x))):
    word_dict_to_replace[sorted_key] = word_dict_to_replace_unsorted[sorted_key]

punctuation_to_replace = {
    '？': '?',
}

alphabets_to_replace = {
    'a': '에이',
    'A': '에이',
    'b': '비',
    'B': '비',
    'c': '씨',
    'C': '씨',
    'd': '디',
    'D': '디',
    'e': '이',
    'E': '이',
    'f': '에프',
    'F': '에프',
    'g': '지',
    'G': '지',
    'h': '에이치',
    'H': '에이치',
    'i': '아이',
    'I': '아이',
    'j': '제이',
    'J': '제이',
    'k': '케이',
    'K': '케이',
    'l': '엘',
    'L': '엘',
    'm': '엠',
    'M': '엠',
    'n': '엔',
    'N': '엔',
    'o': '오',
    'O': '오',
    'p': '피',
    'P': '피',
    'q': '큐',
    'Q': '큐',
    'r': '알',
    'R': '알',
    's': '에스',
    'S': '에스',
    't': '티',
    'T': '티',
    'u': '유',
    'U': '유',
    'v': '비',
    'V': '비',
    'w': '더블유',
    'W': '더블유',
    'x': '엑스',
    'X': '엑스',
    'y': '와이',
    'Y': '와이',
    'z': '지',
    'Z': '지',
}

numbers_to_replace = {
    '1': '일',
    '2': '이',
    '3': '삼',
    '4': '사',
    '5': '오',
    '6': '육',
    '7': '칠',
    '8': '팔',
    '9': '구',
}

ETRI_P = re.compile('[(][^(^)]+[)]/[(][^(^)]+[)]')

# Written to Spoken
def get_etri_w2s_pairs(script):
    results = ETRI_P.findall(script)
    pairs_dict = dict()
    for result in results:
        written, spoken = result.split(')/(')
        pairs_dict[written[1:]] = spoken[0:-1]
    return pairs_dict

def cleanse_etri_w2s_pairs(script):
    results = ETRI_P.findall(script)
    pairs_dict = dict()
    for result in results:
        written, spoken = result.split(')/(')
        pairs_dict[written[1:]] = spoken[0:-1]
    return pairs_dict

def cleanse_etri_w2s_pairs(script):
    results = ETRI_P.findall(script)
    for result in results:
        written, spoken = result.split(')/(')
        script = script.replace(result, spoken[0:-1])
    return script

def get_etri_dict_to_replace(total_pairs_dict):

    '''
    규칙 0. 띄어쓰기로 시작하지 않는다면, 앞에 무조건 띄어쓰기를 추가한다.
    '1조각' -> '한 조각', ' 1조각' -> ' 한 조각'
    규칙 1. 숫자만 있는 변환은 제외한다. (.과 ,도 포함)
    규칙 2. 긴 변환부터 적용한다.
    규칙 3. 순수 영어는 뒤에 띄어쓰기를 추가한다. 
    'KT' -> '케이티', ' KT ' -> ' 케이티 '
    'KTX' -> '케이티X' 변환을 예방
    규칙 4. 1글자짜리 단어는 제외한다.
    '''

    not_only_num_p = re.compile('[^0-9 \t\n\r\f\v,.]')
    korean_pattern = re.compile('[ㄱ-ㅎ|ㅏ-ㅣ|가-힣|.?!]')

    etri_dict_to_replace = dict()

    keys_sorted = list(reversed(sorted(total_pairs_dict.keys(), key = lambda x: len(x))))

    removed_keys = list()

    for key in keys_sorted:
        if not not_only_num_p.search(key):
            removed_keys.append(key)

    for key in keys_sorted:
        if key in removed_keys:
            continue
            
        clean_key = key.strip()
        
        if len(clean_key) == 1:
            continue
        
        clean_value = total_pairs_dict[key].strip()
        
        if not korean_pattern.match(clean_key[-1]):
            etri_dict_to_replace[' ' + clean_key + ' '] = ' ' + clean_value + ' '
        else:
            etri_dict_to_replace[' ' + clean_key] = ' ' + clean_value

    return etri_dict_to_replace

def cleanse_korean(script, custom_dict = None):

    script = ' ' + script

    script = cleanse_etri_w2s_pairs(script)

    for symbol in noise_symbol_list_to_remove:
        script = script.replace(symbol, '')
            
    for key_word in interjection_dict_to_replace:
        script = script.replace(key_word, interjection_dict_to_replace[key_word])

    for key_word in punctuation_to_replace:
        script = script.replace(key_word, punctuation_to_replace[key_word])
          
    for key_word in word_dict_to_replace:
        script = script.replace(key_word, word_dict_to_replace[key_word])

    for key_word in word_dict_to_replace:
        script = script.replace(key_word, word_dict_to_replace[key_word])

    if custom_dict is not None:
        for key_word in custom_dict:
            script = script.replace(key_word, custom_dict[key_word])
    
    for key_word in alphabets_to_replace:
        script = script.replace(key_word, alphabets_to_replace[key_word])

    for key_word in numbers_to_replace:
        script = script.replace(key_word, numbers_to_replace[key_word])
        
    for symbol in symbol_list_to_remove:
        script = script.replace(symbol, '')

    script = script.replace('  ', ' ')

    script = script.strip()

    return script