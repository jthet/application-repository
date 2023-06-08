# Image: jthet/fill_mask_app
#
FROM python:3.10

WORKDIR /app

RUN pip install transformers==4.29.2 
RUN pip3 install torch==2.0.1 --default-timeout=1000
# Having issues :
# WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.org', port=443): Read timed out. (read timeout=15)")': /simple/tensorflow/
# and
# WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out. (read timeout=15)")': /packages/3f/b2/33372601ed71fb41049642f8f6e1e142215e8b5c3463df434fc8885db278/tensorflow-2.12.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#
# is there a way to just import transformers.pipeline?
# RUN pip install torch==2.0.1

COPY unmasker.py .

CMD [ "python3", "./unmasker.py" ]