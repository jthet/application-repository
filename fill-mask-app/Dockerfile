# Image: jthet/fill_mask_app
# Run with:
#   docker run -it --rm jthet/fill_mask_app:latest <"sentance with <mask>"> <number-of-returns>


FROM huggingface/transformers-pytorch-cpu:4.18.0

WORKDIR /app

COPY unmasker.py /app

ENTRYPOINT [ "python3", "./unmasker.py" ]



