FROM python:3.12-slim

WORKDIR /app

RUN useradd -m -u 1000 user && mkdir -p /home/user/.cache/indomeme-xai && chown -R user:user /app /home/user

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .
RUN chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    INDOMEME_DEVICE=cpu \
    INDOMEME_CACHE_DIR=/home/user/.cache/indomeme-xai

EXPOSE 7860

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
