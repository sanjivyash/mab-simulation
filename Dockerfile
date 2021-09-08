FROM docker.cse.iitb.ac.in/cs747

RUN mkdir -p /cs747

WORKDIR /cs747

COPY . .

WORKDIR /cs747/submission

CMD ["python", "verifyOutput.py"]
