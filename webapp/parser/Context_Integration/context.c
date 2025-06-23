#include <stdio.h>
#include <sqlite3.h>

int main() {
    sqlite3* db;
    char* errorMessage = NULL;

    // Open database
    if (sqlite3_open("context_elections.db", &db)) {
        fprintf(stderr, "Error opening database: %s\n", sqlite3_errmsg(db));
        return 1;
    }

    // Create table
    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);";
    if (sqlite3_exec(db, createTableSQL, nullptr, nullptr, &errorMessage) != SQLITE_OK) {
        fprintf(stderr, "Error creating table: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    }

    // Insert data
    const char* insertSQL = "INSERT INTO users (name) VALUES ('Alice');";
    if (sqlite3_exec(db, insertSQL, nullptr, nullptr, &errorMessage) != SQLITE_OK) {
        fprintf(stderr, "Error inserting data: %s\n", errorMessage);
        sqlite3_free(errorMessage);
    }

    // Close database
    sqlite3_close(db);
    printf("Database operations completed successfully.\n");

    return 0;
}
