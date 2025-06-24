using System;
using Npgsql;

class Program
{
    static void Main()
    {
        // Load connection string from environment variable or config file for security
        string connString = Environment.GetEnvironmentVariable("PG_CONN_STRING");
        if (string.IsNullOrEmpty(connString))
        {
            Console.WriteLine("Error: PostgreSQL connection string not set. Please set the PG_CONN_STRING environment variable.");
            return;
        }

        using (var conn = new NpgsqlConnection(connString))
        {
            try
            {
                conn.Open();

                // Example: Create table if not exists (warehouse_election_results)
                string createTableSQL = @"CREATE TABLE IF NOT EXISTS warehouse_election_results (
                    id SERIAL PRIMARY KEY,
                    batch_id UUID NOT NULL,
                    state TEXT,
                    county TEXT,
                    contest_title TEXT,
                    candidate TEXT,
                    party TEXT,
                    votes INTEGER,
                    precinct TEXT,
                    election_date DATE,
                    processed_at TIMESTAMP DEFAULT NOW()
                );";
                using (var command = new NpgsqlCommand(createTableSQL, conn))
                {
                    command.ExecuteNonQuery();
                }

                // Example: Insert data
                string insertSQL = @"INSERT INTO warehouse_election_results (batch_id, state, county, contest_title, candidate, party, votes, precinct, election_date) VALUES (@batch_id, @state, @county, @contest_title, @candidate, @party, @votes, @precinct, @election_date);";
                using (var command = new NpgsqlCommand(insertSQL, conn))
                {
                    command.Parameters.AddWithValue("batch_id", Guid.NewGuid());
                    command.Parameters.AddWithValue("state", "NY");
                    command.Parameters.AddWithValue("county", "Rockland");
                    command.Parameters.AddWithValue("contest_title", "Presidential Election");
                    command.Parameters.AddWithValue("candidate", "Alice");
                    command.Parameters.AddWithValue("party", "Independent");
                    command.Parameters.AddWithValue("votes", 1234);
                    command.Parameters.AddWithValue("precinct", "Precinct 1");
                    command.Parameters.AddWithValue("election_date", DateTime.UtcNow.Date);
                    command.ExecuteNonQuery();
                }

                Console.WriteLine("Database operations completed successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error: " + ex.Message);
            }
        }
    }
}