# Installation and Running Guide for Elasticsearch on Windows

## Step 1 - Download Elasticsearch (8.15.0)

1. Download the Elasticsearch zip file for Windows from the [Elasticsearch website](https://www.elastic.co/downloads/elasticsearch).
2. Extract the Elasticsearch folder and place it in the `C:` or `D:` drive. For example: `C:\elasticsearch-8.15.0`.

## Step 2 - Run Elasticsearch as a Background Service on Windows

1. Open **Command Prompt** and navigate to the extracted Elasticsearch folder.
2. Run the following command to install Elasticsearch as a Windows service:

   ```bash
   .\bin\elasticsearch-service.bat install
   ```

3. Run the following command to start the Elasticsearch service:

   ```bash
   .\bin\elasticsearch-service.bat start
   ```

4. To stop the Elasticsearch service, run the following command:

   ```bash
   .\bin\elasticsearch-service.bat stop
   ```

5. To uninstall the Elasticsearch service, run the following command:

   ```bash
   .\bin\elasticsearch-service.bat remove
   ```

## Step 3 - Reset the `elastic` User Password

1. Open **Command Prompt** in the Elasticsearch folder.
2. Run the following command to reset the `elastic` user's password:

   ```bash
   .\bin\elasticsearch-reset-password.bat -i -u elastic
   ```

3. Follow the on-screen instructions to set a new password for the `elastic` user.

## Step 4 - Verify Elasticsearch is Running

1. Open a web browser and go to the following URL:

   ```
   http://localhost:9200
   ```

2. Enter the username and password for the `elastic` account.
3. If Elasticsearch is running, detailed information about Elasticsearch will be displayed in the browser.

## [Optional] Set Memory Limits for Elasticsearch on Windows

1. By default, only 1 GB of memory is allocated for the Elasticsearch service, which may cause errors when running many queries or handling large datasets.
2. Open **Command Prompt** in the Elasticsearch folder.
3. Run the following command to open the service configuration window:

   ```bash
   .\bin\elasticsearch-service.bat manager
   ```

4. In the configuration window, go to the **Java** tab and set the values for `Initial memory pool` and `Maximum memory pool` to a higher value, such as 10 GB (equivalent to 10240 MB).