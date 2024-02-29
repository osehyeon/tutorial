import xgboost as xgb
import inspect

# xgboost의 callback 모듈을 가져와서 소스 코드를 문자열로 저장합니다.
callback_source = inspect.getsource(xgb.callback)

# _fmt_metric 함수가 소스 코드에 포함되어 있는지 확인합니다.
if '_fmt_metric' in callback_source:
    print("'_fmt_metric' 함수가 존재합니다.")
else:
    print("'_fmt_metric' 함수가 존재하지 않습니다.")