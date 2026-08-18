libs = ['fastapi', 'uvicorn', 'sklearn', 'catboost', 'torch', 'tensorflow', 'pandas', 'numpy', 'pydantic']
for lib in libs:
    try:
        __import__(lib)
        print(f"{lib}: Installed")
    except Exception as e:
        print(f"{lib}: NOT Installed / Error: {e}")
