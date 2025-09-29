cd webapi
docker build --no-cache -t my-python-api .
docker run --rm --name webapi -p 8000:8000 my-python-api
Open http://localhost:8000/app/