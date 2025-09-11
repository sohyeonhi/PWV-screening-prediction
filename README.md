기본 건강검진·생활습관 데이터만으로 ‘맥파전달속도(PWV) 추가검사 필요성’을 예측하는 딥러닝 기반 분류 모델 프로젝트입니다. 모델 성능뿐 아니라, 재현성·해석 가능성·서비스 이용성을 함께 고려해 설계했습니다.

리포지토리 구조
- preprocess.py : 원천데이터 전처리, 파생변수 생성
- feature_select.py : 피처 안정성 평가, 선정 로직
- train_model.py : 모델 학습, 검증, 조기중단 
- mlp_eva.py : 모델 평가, 임계값 
- mlp_main.py : 메인 실행부 
- final_features0.3.csv : 최종 선정 피처 목록
- decision.json : 피처 조합 선택 근거
- optuna_best_params8.json : 하이퍼파라미터 탐색 결과
