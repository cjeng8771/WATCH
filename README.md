# WATCH: A Distributed Clock Time Offset Estimation Tool for Software-Defined Radio Platforms

WATCH uses linear algebra methods to estimate the difference between each node's local clock in relation to one another, therefore checking if the platform's nodes are time synchronized. This is done by analyzing data from repeated transmissions between nodes.

The script runs through generating an IQ file, setting up an experiment, running the experiment, and analyzing the data with WATCH post-processing. There is an option to skip to post-processing if a researcher has previously collected data they want to analyze.

This tool was originally developed using Shout on the Platform for Open Wireless Data-driven Experimental Research (POWDER), but can be adapted and used on other software-defined radio platforms for checking time synchronization.

## Before beginning: download [watch.py](https://github.com/cjeng8771/WATCH/blob/main/watch.py) into a new directory on the local host.

## Option (1): Full Experiment & WATCH Post-Processing
To run through the full experiment including data collection and WATCH:
1. Run `python watch.py` on local host.
2. Configuration
    * Choose [y/n] for debug. [y] will display print statements for all intermediate steps. [n] will display only important print statements and final results. 
    * Choose [1] to run the entire program.
3. Create Message to Transmit
    * To transmit plain text [p]: create a message that is 94 characters long. 
    * To transmit PN codes [r]: determine which initial states and taps to use for in-phase and quadrature. Specify these when prompted or hit enter without entering text to use the default states and taps.
    * Choose a file name for the created IQ file and enter it when prompted.
4. Resource Reservation with the [Platform for Open Wireless Data-driven Experimental Research (POWDER)](https://powderwireless.net/) and Initializing Node Sessions
    * Follow all instructions printed to STDOUT.
    * **Note:** List View refers to the second tab in the table at the bottom of the screen when viewing the active ready experiment.
5. Defining Experiment Nodes
    * Once all nodes are ready again, copy the `[username]@[orch-node-name]` from the **SSH command** column and **orch** row in the table.
    * Record the node numbers for all the non-orch nodes. These are part of the node-name in the **SSH command** column for each node. Enter these when prompted.
    * Record the node names for all the non-orch nodes. These are part of the node ID in the **ID** column for each node. Enter these, in all lowercase and abbreviated the same way they are in the column, when prompted.
6. Configuring Nodes for Transmission
    * Follow all instructions printed to STDOUT that describe how to set up the node SSH sessions for the experiment. After continuing, the script will secure copy the IQ file previously created to all nodes and check that the `meascli.py` script on each node is enabling the use of the external clock.
    * When prompted about modifications to the experiment JSON file, hit enter to keep the default parameter or type the desired modification in the same format as the default is shown. 
    * **Note:** Make sure to choose a txfreq/rxfreq in the experiment's reserved range.
    * **Note:** Choose rxrepeat based on how many transmission iterations are desired for each link.
    * Record the full ID listed in the **ID** column of the experiment table for all comp nodes. When prompted, enter these in the script as directed. The script will use these IDs to secure copy the modified JSON file to all comp nodes for the experiment.
7. Running the Experiment
    * Follow all instructions printed to STDOUT. These will walk through testing preparation for the experiment and starting data collection with Shout. 
    * **Note:** The Shout measurement framework is used to automate TX and RX functions across multiple nodes in the POWDER network. 
    * **Note:** When the second orch SSH session returns to the command prompt, the script will confirm there is a Shout data collection folder on the remote host and then secure copy it to the local host.
8. Offset Estimation Post-Processing with WATCH
    * **NOTE:** The PSD plots can be informational to observe, but depending on the number of nodes and iterations in the experiment, there can be a large number of output plots to handle before continuing with the analysis. If DEBUG was enabled at the beginning, these will print despite what is chosen at this step.
    * **NOTE:** The weighted least squares method can be invoked, but the use of each link's signal to noise ratio (SNR) is not optimized. The results will not differ much from those without the weighted least squares method.
    * **Final WATCH analysis results will be printed, by iteration, to STDOUT.**
    * WATCH reports an estimate local clock offset at each node, in comparison to one another, and with the first node as a reference for time zero, by cross correlating the received packet from the Shout data with the transmitted packet to find each node's offset index. Offset index is the length, in number of samples, between the local clock time zero and when the receiver actually receives the first sample of the transmitted packet.
    * The least square error and root mean squared error (RMSE) included in the final results compare each link's true found offset indicies to estimated offsets that are calculated using the WATCH algorithm results.

## Option (2): WATCH Post-Processing with Previous Data
To run through analysis with WATCH for previously collected data:
1. Ensure the previously collected Shout data folder, in the format `Shout_meas_MM-DD-YYYY_HH-MM-SS`, and the associated transmitted IQ file, are in the same working directory as `watch.py` on the local host.
2. Run `python watch.py` on local host.
3. Configuration
    * Choose [y/n] for debug. [y] will display print statements for all intermediate steps. [n] will display only important print statements and final results. 
    * Choose [2] to run just WATCH post-processing.
4. Enter the name of the Shout data folder and IQ transmitted file when prompted.
5. Offset Estimation Post-Processing with WATCH
    * **NOTE:** The PSD plots can be informational to observe, but depending on the number of nodes and iterations in the experiment, there can be a large number of output plots to handle before continuing with the analysis. If DEBUG was enabled at the beginning, these will print despite what is chosen at this step.
    * **NOTE:** The weighted least squares method can be invoked, but the use of each link's signal to noise ratio (SNR) is not optimized. The results will not differ much from those without the weighted least squares method.
    * **Final WATCH analysis results will be printed, by iteration, to STDOUT.**
    * WATCH reports an estimate local clock offset at each node, in comparison to one another, and with the first node as a reference for time zero, by cross correlating the received packet from the Shout data with the transmitted packet to find each node's offset index. Offset index is the length, in number of samples, between the local clock time zero and when the receiver actually receives the first sample of the transmitted packet.
    * The least square error and root mean squared error (RMSE) included in the final results compare each link's true found offset indicies to estimated offsets that are calculated using the WATCH algorithm results.
