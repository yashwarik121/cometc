# Aster & Row - Orders Data Dictionary

This document defines the schema for the `orders.json` dataset. Some fields contain highly sensitive personal or business information and are marked as **INTERNAL**. These fields must never be exposed to public interfaces, customers, or third-party systems without explicit authorization.

## Schema Definition

| Field Name | Type | Description | Visibility / Sensitivity |
|---|---|---|---|
| `order_id` | String | Unique identifier for the order, format 'AR-XXXXX' (5 digits). | Public |
| `customer_name` | String | Full name of the customer who placed the order. | Public (to customer) |
| `customer_email` | String | Email address used for order communications. | **INTERNAL - Must never be exposed** |
| `shipping_address` | Object | Delivery address containing `street`, `city`, `state`, `zip`, and `country`. | **INTERNAL - Must never be exposed** |
| `items` | Array | List of products purchased in the order. Each item is an object with `product_name`, `quantity`, and `unit_price`. | Public |
| `order_total` | Number | The total monetary value of the order. | Public |
| `order_date` | String | ISO 8601 formatted date and time when the order was placed. | Public |
| `status` | String | Current state of the order. Valid values: `processing`, `shipped`, `delivered`, `cancelled`, `returned`, `return_requested`. | Public |
| `tracking_number` | String/Null | Shipping carrier's tracking identifier, if available. | Public |
| `carrier` | String/Null | Name of the shipping carrier (e.g., UPS, FedEx, USPS). | Public |
| `estimated_delivery` | String/Null | ISO 8601 formatted date of the estimated delivery. | Public |
| `actual_delivery` | String/Null | ISO 8601 formatted date of the actual delivery. | Public |
| `internal_notes` | String | Notes entered by customer service or fulfillment teams regarding the customer or order. | **INTERNAL - Must never be exposed** |
| `risk_score` | Number | A computed fraud risk score ranging from 0 to 100. | **INTERNAL - Must never be exposed** |
| `payment_method` | String | Masked payment method information (e.g., "Visa **** 4242"). | Public |
| `cancellation_reason` | String/Null | Reason for order cancellation, if applicable. | Public (with care) |
| `return_reason` | String/Null | Reason for returning an item, provided by the customer. | Public |
