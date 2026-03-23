FROM node:20-alpine3.22

WORKDIR /app

VOLUME [ "/data" ]

COPY package*.json ./

RUN npm install

COPY . .

EXPOSE 8000

CMD ["npm", "run", "dev"]