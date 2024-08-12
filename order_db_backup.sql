-- MySQL dump 10.13  Distrib 8.0.37, for Linux (x86_64)
--
-- Host: 172.30.1.90    Database: order_db
-- ------------------------------------------------------
-- Server version	8.0.37-0ubuntu0.22.04.3

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `inventory_management_table`
--

DROP TABLE IF EXISTS `inventory_management_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `inventory_management_table` (
  `flavor1_status` int NOT NULL DEFAULT '0',
  `flavor2_status` int NOT NULL DEFAULT '0',
  `flavor3_status` int NOT NULL DEFAULT '0',
  `topping1_status` int NOT NULL DEFAULT '0',
  `topping2_status` int NOT NULL DEFAULT '0',
  `topping3_status` int NOT NULL DEFAULT '0',
  `flavor1` int NOT NULL DEFAULT '0',
  `flavor2` int NOT NULL DEFAULT '0',
  `flavor3` int NOT NULL DEFAULT '0',
  `topping1` int NOT NULL DEFAULT '0',
  `topping2` int NOT NULL DEFAULT '0',
  `topping3` int NOT NULL DEFAULT '0',
  `date_time` timestamp NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `inventory_management_table`
--

LOCK TABLES `inventory_management_table` WRITE;
/*!40000 ALTER TABLE `inventory_management_table` DISABLE KEYS */;
INSERT INTO `inventory_management_table` VALUES (199,198,198,199,198,198,1,2,2,1,2,2,'2024-07-31 03:17:15');
/*!40000 ALTER TABLE `inventory_management_table` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `order_management_table`
--

DROP TABLE IF EXISTS `order_management_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `order_management_table` (
  `order_ID` int NOT NULL AUTO_INCREMENT,
  `date_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `choco_order` int NOT NULL DEFAULT '0',
  `vanila_order` int NOT NULL DEFAULT '0',
  `strawberry_order` int NOT NULL DEFAULT '0',
  `topping1_order` int NOT NULL DEFAULT '0',
  `topping2_order` int NOT NULL DEFAULT '0',
  `topping3_order` int NOT NULL DEFAULT '0',
  `weather` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`order_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `order_management_table`
--

LOCK TABLES `order_management_table` WRITE;
/*!40000 ALTER TABLE `order_management_table` DISABLE KEYS */;
INSERT INTO `order_management_table` VALUES (1,'2024-07-31 03:18:00',0,2,0,0,2,0,'서울 현재 날씨: 31°C 튼구름'),(2,'2024-07-31 03:18:12',0,2,0,0,2,0,'서울 현재 날씨: 31°C 튼구름'),(3,'2024-07-31 03:18:32',99,9,9,3,3,9,NULL),(4,'2024-07-31 03:18:42',0,2,0,0,2,0,'서울 현재 날씨: 31°C 튼구름');
/*!40000 ALTER TABLE `order_management_table` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `purchase_record_table`
--

DROP TABLE IF EXISTS `purchase_record_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `purchase_record_table` (
  `user_ID` int NOT NULL AUTO_INCREMENT,
  `choco` int NOT NULL DEFAULT '0',
  `vanila` int NOT NULL DEFAULT '0',
  `strawberry` int NOT NULL DEFAULT '0',
  `choco_count` int NOT NULL DEFAULT '0',
  `vanila_count` int NOT NULL DEFAULT '0',
  `strawberry_count` int NOT NULL DEFAULT '0',
  `topping1` int NOT NULL DEFAULT '0',
  `topping2` int NOT NULL DEFAULT '0',
  `topping3` int NOT NULL DEFAULT '0',
  `topping1_count` int NOT NULL DEFAULT '0',
  `topping2_count` int NOT NULL DEFAULT '0',
  `topping3_count` int NOT NULL DEFAULT '0',
  PRIMARY KEY (`user_ID`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `purchase_record_table`
--

LOCK TABLES `purchase_record_table` WRITE;
/*!40000 ALTER TABLE `purchase_record_table` DISABLE KEYS */;
INSERT INTO `purchase_record_table` VALUES (1,1,2,2,0,0,0,1,2,2,0,0,0),(2,0,0,0,0,0,0,0,0,0,0,0,0);
/*!40000 ALTER TABLE `purchase_record_table` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `user_info_table`
--

DROP TABLE IF EXISTS `user_info_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `user_info_table` (
  `user_id` int NOT NULL AUTO_INCREMENT,
  `point` int DEFAULT '0',
  `gender` enum('Male','Female') DEFAULT NULL,
  `name` varchar(100) NOT NULL,
  `phone_num` varchar(15) DEFAULT NULL,
  `birthday` date NOT NULL,
  `last_modified` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `photo_path` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_info_table`
--

LOCK TABLES `user_info_table` WRITE;
/*!40000 ALTER TABLE `user_info_table` DISABLE KEYS */;
INSERT INTO `user_info_table` VALUES (1,15,'Male','이성민','01041829952','1995-10-30','2024-07-31 04:23:47',NULL),(2,0,'Male','박정한','01020895910','1999-05-10','2024-07-31 03:11:11',NULL);
/*!40000 ALTER TABLE `user_info_table` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-07-31 16:42:09
