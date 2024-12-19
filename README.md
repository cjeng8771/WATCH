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
    * Choose [y/n] for plots. [y] will display all plots. [n] will display only a few examples of each plot type.
    * Choose [1] to run the entire program.

> [!NOTE]
> Depending on the number of nodes in the experiment, displaying all plots could produce a large number of plots since it will render at least one plot per iteration per link. These will all need to be closed before proceeding with the program.

3. Create Message to Transmit
    * To transmit plain text [p]: Choose a plain text message that is 94 characters long. 
    * To transmit PN codes [r]: Taps used in this program are derived from **Table 2: Some feedback taps for maximal sequence length** in [*Pseudo noise sequences for engineers* (R.N. Mutagi 1996)](https://www.researchgate.net/publication/230813025_Pseudo_Noise_Sequences_for_Engineers_Electronics_and_Communication_Engineering_Journal_UK_Vol8_No2_April_1996).
      - Specify a number of stages, N, from 3 to 25, where the length of the PN code, $L = 2^N - 1$. The program will provide the valid options for taps based on the chosen N number of stages.
      - Choose an initial state vector for the in-phase sequence or press enter to use the shown default.
      - Choose one of the valid options for taps to use for the in-phase sequence or press enter to use the shown default.
      - Choose an initial state vector for the quadrature sequence or press enter to use the shown default.
      - Choose one of the valid options for taps to use for the quadrature sequence or press enter to use the shown default.
    * Choose a file name for the created IQ file and enter it when prompted.

> [!NOTE]
> Information detailing what plain text message/PN code length was transmitted will be recorded in `iqinfo.txt` within the data collection folder from Shout.

4. Follow all instructions printed to STDOUT for Resource Reservation with the [Platform for Open Wireless Data-driven Experimental Research (POWDER)](https://powderwireless.net/) and Initializing Node Sessions.

> [!TIP]
> List View refers to the second tab in the table at the bottom of the screen when viewing the active ready experiment on [powderwireless.net](https://powderwireless.net/).

5. Once all nodes are ready again, copy each listed `[username]@[node-name]` from the **SSH command** column in the table, and enter them, separated by a space, when prompted.
6. Configuring Nodes for Transmission
    * Follow all instructions printed to STDOUT that describe how to set up the node SSH sessions for the experiment. After continuing, the script will secure copy the IQ file previously created to the `/local/repository/shout/signal_library` directory in all nodes and check that the `meascli.py` script in the `/local/repository/shout` directory on each node is enabling the use of the external clock and appropriate port. 
    * When prompted about modifications to the experiment JSON file, hit enter to keep the default parameter or type the desired modification in the same format as the default is shown. 
    * From the **ID** column of the experiment table, record the full ID for every node with a valid ssh command, other than the orchestrator. When prompted, enter these in the script as directed. The script will use these IDs to secure copy the modified JSON file to the `/local/repository/etc/cmdfiles` directory on all nodes for the experiment.

> [!IMPORTANT]
> Make sure to choose a `txfreq`/`rxfreq` in the experiment's reserved range. This range is recommended to be between 3360 MHz and 3450 MHz to avoid interference. Choose `rxrepeat` based on how many transmission iterations are desired for each link.

7. Running the Experiment
    * Follow all instructions printed to STDOUT. These will walk through testing preparation for the experiment and starting data collection with Shout. The Shout measurement framework is used to automate TX and RX functions across multiple nodes in the POWDER network. 

> [!NOTE]
> When the second orch SSH session returns to the command prompt, the script will confirm there is a Shout data collection folder on the remote host and then secure copy it to the local host.

8. Offset Estimation Post-Processing with WATCH
    * WATCH reports an estimate local clock offset at each node, in comparison to one another, with the first node as a reference for time zero, by cross correlating the received packet from the Shout data with the transmitted packet. The offsets are the difference in microseconds between the local clock time zero and when the receiver actually receives the first sample of the transmitted packet.
    * The least square error and root mean squared error (RMSE) included in the final results compare the offset found at each link through cross correlation to the estimated offsets that are back-calculated using the WATCH algorithm results.
    * The weighted least squares method can be invoked to influence the offsets based on the estimated reliability of each link using its SNR.
    * The PSD plots can be informational to observe, but depending on the number of nodes and iterations in the experiment, there can be a large number of output plots to handle before continuing with the analysis. If DEBUG was enabled at the beginning, these will print despite what is chosen at this step.
    * WATCH will prematurely alert if it detects too many links with insufficient data to calculate the node clock offsets, leading to inconclusive results. This signifies that a majority of the links in the experiment did not receive detectable signals, which can be caused by situations such as out of range nodes or use of an inactive port.

> [!IMPORTANT]
> Final WATCH analysis results will be printed, by iteration, and displayed in microseconds ($\mu s$). They are also recorded in `results.txt` in the current working directory.

> [!NOTE]
> Offset results on the order of 10s-100s of $\mu s$ indicate a time-synchronized network. However, results should not be expected to be much less than $\frac{1}{sampleRate}$ (4 $\mu s$ when using the default sample_rate of 250kHz).\
> \
> Offset results on the order of 1000s of $\mu s$ indicate a non time-synchronized network. Delays this large, on the order of milliseconds, show the experiment nodes' local clocks are significantly offset from one another.

## Option (2): WATCH Post-Processing with Previous Data
To run through analysis with WATCH for previously collected data:
1. Ensure the previously collected Shout data folder, in the format `Shout_meas_MM-DD-YYYY_HH-MM-SS`, and the associated transmitted IQ file, are in the same working directory as `watch.py` on the local host.
2. Run `python watch.py` on local host.
3. Configuration
    * Choose [y/n] for debug. [y] will display print statements for all intermediate steps. [n] will display only important print statements and final results. 
    * Choose [y/n] for plots. [y] will display all plots. [n] will display only a few examples of each plot type.
    * Choose [2] to run just WATCH post-processing.

> [!NOTE]
> Depending on the number of nodes in the experiment, displaying all plots could produce a large number of plots since it will render at least one plot per iteration per link. These will all need to be closed before proceeding with the program.

4. Enter the name of the Shout data folder and IQ transmitted file when prompted.
5. Offset Estimation Post-Processing with WATCH
    * WATCH reports an estimate local clock offset at each node, in comparison to one another, with the first node as a reference for time zero, by cross correlating the received packet from the Shout data with the transmitted packet. The offsets are the difference in microseconds between the local clock time zero and when the receiver actually receives the first sample of the transmitted packet.
    * The least square error and root mean squared error (RMSE) included in the final results compare the offset found at each link through cross correlation to the estimated offsets that are back-calculated using the WATCH algorithm results.
    * The weighted least squares method can be invoked to influence the offsets based on the estimated reliability of each link using its SNR.
    * The PSD plots can be informational to observe, but depending on the number of nodes and iterations in the experiment, there can be a large number of output plots to handle before continuing with the analysis. If DEBUG was enabled at the beginning, these will print despite what is chosen at this step.
    * WATCH will prematurely alert if it detects too many links with insufficient data to calculate the node clock offsets, leading to inconclusive results. This signifies that a majority of the links in the experiment did not receive detectable signals, which can be caused by situations such as out of range nodes or use of an inactive port.

> [!IMPORTANT]
> Final WATCH analysis results will be printed, by iteration, and displayed in microseconds ($\mu s$). They are also recorded in `results.txt` in the current working directory.

> [!NOTE]
> Offset results on the order of 10s-100s of $\mu s$ indicate a time-synchronized network. However, results should not be expected to be much less than $\frac{1}{sampleRate}$ (4 $\mu s$ when using the default sample_rate of 250kHz).\
> \
> Offset results on the order of 1000s of $\mu s$ indicate a non time-synchronized network. Delays this large, on the order of milliseconds, show the experiment nodes' local clocks are significantly offset from one another.
