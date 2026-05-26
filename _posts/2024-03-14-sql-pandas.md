---
title: 'Leveraging Pandas to Interact with SQL'
date: 2024-03-14
permalink: /posts/2024/03/sql-pandas/
tags:
  - sql
  - pandas
  - databases
---

Most data work involves going back and forth between SQL databases and Python. You write a query to pull what you need, load it into a DataFrame, do your analysis, maybe write results back. Pandas has built-in support for this workflow, and once you set it up, you rarely need to leave Python to interact with your database.

This post covers the basics: what SQL and Pandas each do well, and how to connect them so you can query, transform, and write data without switching contexts.


SQL
======

SQL (Structured Query Language) is how you talk to relational databases — selecting rows, filtering, joining tables, aggregating. If your data lives in PostgreSQL, MySQL, SQLite, or similar, you're writing SQL to get it out.

Pandas
======

Pandas is a Python library for working with tabular data. Its core data structure, the DataFrame, gives you fast filtering, grouping, reshaping, and plotting — all the stuff that's tedious in raw SQL or plain Python.

Over the years Pandas has added tight integration with SQL databases through `read_sql()`, `read_sql_query()`, and `to_sql()`. You can run a SQL query and get a DataFrame back in one line, or push a DataFrame into a database table just as easily. This means you can use SQL for what it's good at (joins, filtering large tables on the server side) and Pandas for what it's good at (reshaping, plotting, quick exploratory analysis) — all without leaving your Python environment.


    
    