FROM node:20-alpine3.22

WORKDIR /app

VOLUME [ "/data" ]

COPY reauirements.txt ./

RUN npm install

COPY . .

EXPOSE 5000

CMD ["python3", "main.py"]