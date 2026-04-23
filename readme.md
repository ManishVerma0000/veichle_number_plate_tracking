conda activate yolo_env



docker rm -f kafka
docker rm -f zookeeper
docker network rm kafka-net

docker network create kafka-net

docker run -d --name zookeeper --network kafka-net -p 2181:2181 zookeeper

docker run -d --name kafka --network kafka-net -p 9092:9092 -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 confluentinc/cp-kafka:7.5.0