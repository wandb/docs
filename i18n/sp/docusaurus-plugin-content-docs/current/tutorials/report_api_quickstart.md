---
title: "Report API Quickstart"
format: docusaurus-md
---

export const quartoRawHtml =
[`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114554-91x64vpg</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/report-api-quickstart/runs/91x64vpg" target="_blank">adventurous-aardvark-1</a></strong> to <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">https://wandb.ai/noahluna/report-api-quickstart</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/91x64vpg" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/91x64vpg</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>acc</td><td>▁▂▂▄▅▄▄▅▅▅▅▅▆▅▅▆▆▆▇▆▇▇▇▆▆▇█▇█▇▇██▇██▇▇▇█</td></tr><tr><td>loss</td><td>█▇▆▅▅▅▅▄▃▄▃▃▃▂▂▃▃▂▂▂▂▂▂▁▃▂▂▂▂▂▁▁▂▂▁▁▂▁▂▁</td></tr><tr><td>val_acc</td><td>▁▃▄▄▅▄▅▅▅▅▆▆▇▆▆▇▇▇▇▇▇▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇█▇██</td></tr><tr><td>val_loss</td><td>█▇▆▅▅▄▄▃▃▄▃▄▄▄▃▃▂▂▃▃▃▃▃▂▃▂▃▂▁▂▁▁▁▂▁▂▁▁▁▂</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>acc</td><td>3.46682</td></tr><tr><td>loss</td><td>-0.05696</td></tr><tr><td>val_acc</td><td>3.41633</td></tr><tr><td>val_loss</td><td>0.0291</td></tr></table><br/></div></div>
`,`
 View run <strong style="color:#cdcd00">adventurous-aardvark-1</strong> at: <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/91x64vpg" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/91x64vpg</a><br/>Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114554-91x64vpg/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114601-wzja4a3p</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/report-api-quickstart/runs/wzja4a3p" target="_blank">bountiful-badger-2</a></strong> to <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">https://wandb.ai/noahluna/report-api-quickstart</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/wzja4a3p" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/wzja4a3p</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>acc</td><td>▁▂▃▅▄▅▅▅▆▅▆▇▆▆▆▆▇▆▆▇▇▆▇▆▇▇▇█▇▇▇▇▇█▇▇█▇▇█</td></tr><tr><td>loss</td><td>█▆▆▅▅▅▅▃▃▄▃▄▄▃▄▃▃▂▂▃▃▂▂▃▃▂▁▂▁▂▂▂▂▂▂▂▁▁▂▂</td></tr><tr><td>val_acc</td><td>▁▃▄▅▅▅▅▆▆▆▇▅▆▆▆▆▇▆▆▆█▆▇▇▇▇██▇▇███▇▇███▇█</td></tr><tr><td>val_loss</td><td>█▆▅▅▄▄▄▃▃▄▃▃▃▃▃▃▃▃▂▃▂▂▂▂▂▃▂▂▂▁▁▂▂▁▁▂▁▁▁▂</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>acc</td><td>3.24502</td></tr><tr><td>loss</td><td>0.12912</td></tr><tr><td>val_acc</td><td>2.98823</td></tr><tr><td>val_loss</td><td>0.11284</td></tr></table><br/></div></div>
`,`
 View run <strong style="color:#cdcd00">bountiful-badger-2</strong> at: <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/wzja4a3p" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/wzja4a3p</a><br/>Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114601-wzja4a3p/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114608-1mbl3kdo</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/report-api-quickstart/runs/1mbl3kdo" target="_blank">clairvoyant-chipmunk-3</a></strong> to <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">https://wandb.ai/noahluna/report-api-quickstart</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/1mbl3kdo" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/1mbl3kdo</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>acc</td><td>▁▃▃▄▄▄▅▅▆▅▆▆▅▅▆▅▆▆▇▆▆▆▇▆▇▇▇▇▇▇▇▇▇▇▇▇████</td></tr><tr><td>loss</td><td>█▆▄▄▄▄▃▄▃▃▃▃▃▂▃▂▂▃▃▂▃▂▂▂▂▁▂▁▁▂▁▁▁▂▁▂▁▂▂▁</td></tr><tr><td>val_acc</td><td>▁▂▃▄▄▄▄▅▆▅▆▅▆▆▆▆▆▆▇▇▆▇▇▆▆▇▇▇▇██▇▇█▇█▇▇██</td></tr><tr><td>val_loss</td><td>█▆▆▅▄▅▄▃▃▃▃▃▃▃▃▃▃▃▂▃▃▂▃▂▃▂▂▂▂▂▂▂▂▂▁▂▁▂▁▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>acc</td><td>3.40867</td></tr><tr><td>loss</td><td>0.0212</td></tr><tr><td>val_acc</td><td>3.09913</td></tr><tr><td>val_loss</td><td>0.07628</td></tr></table><br/></div></div>
`,`
 View run <strong style="color:#cdcd00">clairvoyant-chipmunk-3</strong> at: <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/1mbl3kdo" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/1mbl3kdo</a><br/>Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114608-1mbl3kdo/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114621-pi23jeyg</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/report-api-quickstart/runs/pi23jeyg" target="_blank">dastardly-duck-4</a></strong> to <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">https://wandb.ai/noahluna/report-api-quickstart</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/pi23jeyg" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/pi23jeyg</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
<style>
    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}
    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }
    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }
    </style>
<div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>acc</td><td>▁▃▃▃▅▅▅▅▅▆▅▆▆▅▅▆▆▇▇▆▇▆▇▇▇▇▇▇█▇▇▇█▇▇█▇▇▇█</td></tr><tr><td>loss</td><td>█▅▄▄▄▄▄▄▃▃▃▃▂▂▃▂▂▂▃▂▂▂▂▂▂▁▂▂▁▂▂▂▁▁▂▂▂▂▁▁</td></tr><tr><td>val_acc</td><td>▁▃▃▄▄▄▄▅▅▅▅▆▆▆▅▆▆▆▇▇▆▇▆▇▆▇▇▇▇▇█▇▇▇▇▇▇█▇█</td></tr><tr><td>val_loss</td><td>█▇▆▅▅▅▄▄▄▃▃▄▃▃▃▄▃▃▂▃▃▂▂▃▂▂▂▂▁▂▂▂▂▁▂▁▂▁▂▂</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>acc</td><td>3.39669</td></tr><tr><td>loss</td><td>0.06379</td></tr><tr><td>val_acc</td><td>3.17357</td></tr><tr><td>val_loss</td><td>0.06125</td></tr></table><br/></div></div>
`,`
 View run <strong style="color:#cdcd00">dastardly-duck-4</strong> at: <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/pi23jeyg" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/pi23jeyg</a><br/>Synced 5 W&B file(s), 0 media file(s), 2 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114621-pi23jeyg/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114628-ntfiqgev</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/report-api-quickstart/runs/ntfiqgev" target="_blank">eloquent-elephant-5</a></strong> to <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/report-api-quickstart" target="_blank">https://wandb.ai/noahluna/report-api-quickstart</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/ntfiqgev" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/ntfiqgev</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">eloquent-elephant-5</strong> at: <a href="https://wandb.ai/noahluna/report-api-quickstart/runs/ntfiqgev" target="_blank">https://wandb.ai/noahluna/report-api-quickstart/runs/ntfiqgev</a><br/>Synced 5 W&B file(s), 2 media file(s), 4 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114628-ntfiqgev/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114638-nia76bdz</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/nia76bdz" target="_blank">northern-flower-1</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/nia76bdz" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/nia76bdz</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">northern-flower-1</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/nia76bdz" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/nia76bdz</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114638-nia76bdz/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114651-rngn3yxh</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/rngn3yxh" target="_blank">comic-disco-2</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/rngn3yxh" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/rngn3yxh</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">comic-disco-2</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/rngn3yxh" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/rngn3yxh</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114651-rngn3yxh/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114706-bfie4p9o</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/bfie4p9o" target="_blank">skilled-smoke-3</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/bfie4p9o" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/bfie4p9o</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">skilled-smoke-3</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/bfie4p9o" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/bfie4p9o</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114706-bfie4p9o/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114720-qryugudj</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/qryugudj" target="_blank">sparkling-paper-4</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/qryugudj" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/qryugudj</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">sparkling-paper-4</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/qryugudj" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/qryugudj</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114720-qryugudj/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114735-qrkp3v0f</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/qrkp3v0f" target="_blank">vital-salad-5</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/qrkp3v0f" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/qrkp3v0f</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">vital-salad-5</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/qrkp3v0f" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/qrkp3v0f</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114735-qrkp3v0f/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114750-3oc6cjmg</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/3oc6cjmg" target="_blank">neat-cherry-6</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/3oc6cjmg" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/3oc6cjmg</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">neat-cherry-6</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/3oc6cjmg" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/3oc6cjmg</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114750-3oc6cjmg/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114806-5y20uxzs</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/5y20uxzs" target="_blank">solar-dream-7</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/5y20uxzs" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/5y20uxzs</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">solar-dream-7</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/5y20uxzs" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/5y20uxzs</a><br/>Synced 5 W&B file(s), 0 media file(s), 11 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114806-5y20uxzs/logs</code>
`,`
wandb version 0.13.10 is available!  To upgrade, please run:
 $ pip install wandb --upgrade
`,`
Tracking run with wandb version 0.13.9
`,`
Run data is saved locally in <code>/Users/noahluna/Documents/GitHub/docodile/docs/tutorials/wandb/run-20230225_114821-pvk8vb2w</code>
`,`
Syncing run <strong><a href="https://wandb.ai/noahluna/lineage-project/runs/pvk8vb2w" target="_blank">jumping-snowball-8</a></strong> to <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">Weights & Biases</a> (<a href="https://wandb.me/run" target="_blank">docs</a>)<br/>
`,`
 View project at <a href="https://wandb.ai/noahluna/lineage-project" target="_blank">https://wandb.ai/noahluna/lineage-project</a>
`,`
 View run at <a href="https://wandb.ai/noahluna/lineage-project/runs/pvk8vb2w" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/pvk8vb2w</a>
`,`
Waiting for W&B process to finish... <strong style="color:green">(success).</strong>
`,`
 View run <strong style="color:#cdcd00">jumping-snowball-8</strong> at: <a href="https://wandb.ai/noahluna/lineage-project/runs/pvk8vb2w" target="_blank">https://wandb.ai/noahluna/lineage-project/runs/pvk8vb2w</a><br/>Synced 5 W&B file(s), 0 media file(s), 5 artifact file(s) and 1 other file(s)
`,`
Find logs at: <code>./wandb/run-20230225_114821-pvk8vb2w/logs</code>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTg3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Quickstart-Report--VmlldzozNjQzOTkz?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/Resizing-panels--VmlldzozNjQzOTk0?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/W-B-Block-Gallery--VmlldzozNjQzOTk1?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/noahluna/report-api-quickstart/reports/W-B-Panel-Gallery--VmlldzozNjQzOTk2?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Report-with-links--VmlldzozMzYzOTMw?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Combined-blocks-report--VmlldzozMzYzOTMy?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Referenced-reports-via-Gallery--VmlldzozMzYzOTMz?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Report-with-markdown--VmlldzozMzYzOTM1?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Parallel-Coordinates-Example--VmlldzozMzYzOTM3?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Parallel-Coordinates-Example-all-in-one---VmlldzozMzYzOTM4?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Adding-tables-to-reports--VmlldzozMzYzOTM5?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Adding-tables-to-reports--VmlldzozMzYzOTM5?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Adding-artifact-lineage-to-reports--VmlldzozMzYzOTQw?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Adding-artifact-lineage-to-reports--VmlldzozMzYzOTQw?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/My-templated-report--VmlldzozMzYzOTQx?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Another-templated-report--VmlldzozMzYzOTQz?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Reinforcement-Learning-Report--VmlldzozMzY0MDQ2?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Weights-Biases-Company--VmlldzozMzY0MDUy?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`,`
<iframe src="https://wandb.ai/megatruong/testing-project-report-api/reports/Untitled-Report--VmlldzozMzY0MDk1?jupyter=true" style="border:none;width:100%;height:1024px;"></iframe>
`];

## What is this? {#what-is-this}

-   Programmatically create and modify reports in Python, including support for editing blocks, panels, and runsets.
-   Create report templates to reuse and share with others

## Quick Links {#quick-links}

-   [🚀 Quickstart Guide](#scrollTo=6QX5y4xx7Ezh) (~5min)
-   [❓ FAQ](#scrollTo=klsDLom6Juc5)
-   [📌 Complete Examples](#scrollTo=56Xtn86jSTOh)

``` python
%%capture
!pip install wandb
```

## Setup {#setup}

``` python
LOG_DUMMY_RUNS = True #@param {type: "boolean"}


import requests
from PIL import Image
from io import BytesIO
import wandb
import pandas as pd
from itertools import product
import random
import math

import wandb
import random
import string

ENTITY = wandb.apis.PublicApi().default_entity
PROJECT = "report-api-quickstart" #@param {type: "string"}
LINEAGE_PROJECT = "lineage-project" #@param {type: "string"}


def get_image(url):
    r = requests.get(url)
    return Image.open(BytesIO(r.content))


def log_dummy_data():
    run_names = [
        "adventurous-aardvark-1",
        "bountiful-badger-2",
        "clairvoyant-chipmunk-3",
        "dastardly-duck-4",
        "eloquent-elephant-5",
        "flippant-flamingo-6",
        "giddy-giraffe-7",
        "haughty-hippo-8",
        "ignorant-iguana-9",
        "jolly-jackal-10",
        "kind-koala-11",
        "laughing-lemur-12",
        "manic-mandrill-13",
        "neighbourly-narwhal-14",
        "oblivious-octopus-15",
        "philistine-platypus-16",
        "quant-quail-17",
        "rowdy-rhino-18",
        "solid-snake-19",
        "timid-tarantula-20",
        "understanding-unicorn-21",
        "voracious-vulture-22",
        "wu-tang-23",
        "xenic-xerneas-24",
        "yielding-yveltal-25",
        "zooming-zygarde-26",
    ]

    opts = ["adam", "sgd"]
    encoders = ["resnet18", "resnet50"]
    learning_rates = [0.01]
    for (i, run_name), (opt, encoder, lr) in zip(
        enumerate(run_names), product(opts, encoders, learning_rates)
    ):
        config = {
            "optimizer": opt,
            "encoder": encoder,
            "learning_rate": lr,
            "momentum": 0.1 * random.random(),
        }
        displacement1 = random.random() * 2
        displacement2 = random.random() * 4
        with wandb.init(
            entity=ENTITY, project=PROJECT, config=config, name=run_name
        ) as run:
            for step in range(1000):
                wandb.log(
                    {
                        "acc": 0.1
                        + 0.4
                        * (
                            math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            + random.random()
                            + displacement1
                            + random.random() * run.config.momentum
                        ),
                        "val_acc": 0.1
                        + 0.4
                        * (
                            math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            - random.random()
                            + displacement1
                        ),
                        "loss": 0.1
                        + 0.08
                        * (
                            3.5
                            - math.log(1 + step + random.random())
                            + random.random() * run.config.momentum
                            + random.random()
                            + displacement2
                        ),
                        "val_loss": 0.1
                        + 0.04
                        * (
                            4.5
                            - math.log(1 + step + random.random())
                            + random.random() * run.config.learning_rate
                            - random.random()
                            + displacement2
                        ),
                    }
                )

    with wandb.init(
        entity=ENTITY, project=PROJECT, config=config, name=run_names[i + 1]
    ) as run:
        img = get_image(
            "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
        )
        image = wandb.Image(img)
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4],
                "float": [1.2, 2.3, 3.4, 4.5],
                "str": ["a", "b", "c", "d"],
                "img": [image] * 4,
            }
        )
        run.log({"img": image, "my-table": df})


class Step:
    def __init__(self, j, r, u, o, at=None):
        self.job_type = j
        self.runs = r
        self.uses_per_run = u
        self.outputs_per_run = o
        self.artifact_type = at if at is not None else "model"
        self.artifacts = []


def create_artifact(name: str, type: str, content: str):
    art = wandb.Artifact(name, type)
    with open("boom.txt", "w") as f:
        f.write(content)
    art.add_file("boom.txt", "test-name")

    img = get_image(
        "https://www.akc.org/wp-content/uploads/2017/11/Shiba-Inu-standing-in-profile-outdoors.jpg"
    )
    image = wandb.Image(img)
    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [1.2, 2.3, 3.4, 4.5],
            "str": ["a", "b", "c", "d"],
            "img": [image] * 4,
        }
    )
    art.add(wandb.Table(dataframe=df), "dataframe")
    return art


def log_dummy_lineage():
    pipeline = [
        Step("dataset-generator", 1, 0, 3, "dataset"),
        Step("trainer", 4, (1, 2), 3),
        Step("evaluator", 2, 1, 3),
        Step("ensemble", 1, 1, 1),
    ]
    for (i, step) in enumerate(pipeline):
        for _ in range(step.runs):
            with wandb.init(project=LINEAGE_PROJECT, job_type=step.job_type) as run:
                # use
                uses = step.uses_per_run
                if type(uses) == tuple:
                    uses = random.choice(list(uses))

                if i > 0:
                    prev_step = pipeline[i - 1]
                    input_artifacts = random.sample(prev_step.artifacts, uses)
                    for a in input_artifacts:
                        run.use_artifact(a)
                # log output artifacts
                for j in range(step.outputs_per_run):
                    # name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
                    name = f"{step.artifact_type}-{j}"
                    content = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=12)
                    )
                    art = create_artifact(name, step.artifact_type, content)
                    run.log_artifact(art)
                    art.wait()

                    # save in pipeline
                    step.artifacts.append(art)

if LOG_DUMMY_RUNS:
  log_dummy_data()
  log_dummy_lineage()
```

``` text
wandb: Currently logged in as: noahluna. Use `wandb login --relogin` to force relogin
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[0] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[1] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[2] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[3] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[4] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[5] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[6] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[7] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[8] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[9] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016750798599484067, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[10] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[11] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[12] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[13] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[14] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[15] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[16] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[17] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[18] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[19] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016807794435104977, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[20] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[21] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[22] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[23] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[24] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[25] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[26] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[27] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[28] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[29] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016711604866820075, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[30] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[31] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[32] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[33] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[34] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[35] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[36] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[37] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[38] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[39] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.01675304931510861, max=1.0)…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[40] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[41] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[42] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[43] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[44] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[45] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[46] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[47] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[48] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016752711116957166, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[49] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[50] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[51] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[52] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[53] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[54] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[55] }} />

``` text
VBox(children=(Label(value='1.412 MB of 1.412 MB uploaded (0.000 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[56] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[57] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016772169433534146, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[58] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[59] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[60] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[61] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[62] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[63] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[64] }} />

``` text
VBox(children=(Label(value='1.412 MB of 1.412 MB uploaded (0.265 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[65] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[66] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016726129164453596, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[67] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[68] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[69] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[70] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[71] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[72] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[73] }} />

``` text
VBox(children=(Label(value='1.412 MB of 1.412 MB uploaded (1.399 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[74] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[75] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.01669719166820869, max=1.0)…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[76] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[77] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[78] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[79] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[80] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[81] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[82] }} />

``` text
VBox(children=(Label(value='1.342 MB of 1.342 MB uploaded (1.134 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[83] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[84] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016709711119377364, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[85] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[86] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[87] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[88] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[89] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[90] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[91] }} />

``` text
VBox(children=(Label(value='1.342 MB of 1.342 MB uploaded (1.329 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[92] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[93] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.01676261458390703, max=1.0)…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[94] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[95] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[96] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[97] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[98] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[99] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[100] }} />

``` text
VBox(children=(Label(value='1.342 MB of 1.342 MB uploaded (1.329 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[101] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[102] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016753528465051203, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[103] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[104] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[105] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[106] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[107] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[108] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[109] }} />

``` text
VBox(children=(Label(value='1.342 MB of 1.342 MB uploaded (1.329 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[110] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[111] }} />

``` text
VBox(children=(Label(value='Waiting for wandb.init()...\r'), FloatProgress(value=0.016752424999140203, max=1.0…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[112] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[113] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[114] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[115] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[116] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[117] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[118] }} />

``` text
VBox(children=(Label(value='0.586 MB of 0.586 MB uploaded (0.573 MB deduped)\r'), FloatProgress(value=1.0, max…
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[119] }} />

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[120] }} />

# 🚀 Quickstart! {#quickstart}

``` python
import wandb.apis.reports as wr
```

``` text
wandb: Thanks for trying out the Report API!
wandb: For a tutorial, check out https://colab.research.google.com/drive/1CzyJx1nuOS4pdkXa2XPaRQyZdmFmLmXV
wandb: 
wandb: Try out tab completion to see what's available.
wandb:   ∟ everything:    `wr.<tab>`
wandb:       ∟ panels:    `wr.panels.<tab>`
wandb:       ∟ blocks:    `wr.blocks.<tab>`
wandb:       ∟ helpers:   `wr.helpers.<tab>`
wandb:       ∟ templates: `wr.templates.<tab>`
wandb:       
wandb: For bugs/feature requests, please create an issue on github: https://github.com/wandb/wandb/issues
```

## Create, save, and load reports {#create-save-and-load-reports}

-   NOTE: Reports are not saved automatically to reduce clutter. Explicitly save the report by calling `report.save()`

``` python
report = wr.Report(
    project=PROJECT,
    title='Quickstart Report',
    description="That was easy!"
)                                 # Create
report.save()                     # Save
wr.Report.from_url(report.url)    # Load
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[121] }} />

## Add content via blocks {#add-content-via-blocks}

-   Use blocks to add content like text, images, code, and more
-   See `wr.blocks` for all available blocks

``` python
report.blocks = [
    wr.TableOfContents(),
    wr.H1("Text and images example"),
    wr.P("Lorem ipsum dolor sit amet. Aut laborum perspiciatis sit odit omnis aut aliquam voluptatibus ut rerum molestiae sed assumenda nulla ut minus illo sit sunt explicabo? Sed quia architecto est voluptatem magni sit molestiae dolores. Non animi repellendus ea enim internos et iste itaque quo labore mollitia aut omnis totam."),
    wr.Image('https://api.wandb.ai/files/telidavies/images/projects/831572/8ad61fd1.png', caption='Craiyon generated images'),
    wr.P("Et voluptatem galisum quo facilis sequi quo suscipit sunt sed iste iure! Est voluptas adipisci et doloribus commodi ab tempore numquam qui tempora adipisci. Eum sapiente cupiditate ut natus aliquid sit dolor consequatur?"),
]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[122] }} />

## Add charts and more via Panel Grid {#add-charts-and-more-via-panel-grid}

-   `PanelGrid` is a special type of block that holds `runsets` and `panels`
    -   `runsets` organize data logged to W&B
    -   `panels` visualize runset data. For a full set of panels, see `wr.panels`

``` python
pg = wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY, PROJECT, "First Run Set"),
        wr.Runset(ENTITY, PROJECT, "Elephants Only!", query="elephant"),
    ],
    panels=[
        wr.LinePlot(x='Step', y=['val_acc'], smoothing_factor=0.8),
        wr.BarPlot(metrics=['acc']),
        wr.MediaBrowser(media_keys='img', num_columns=1),
        wr.RunComparer(diff_only='split', layout={'w': 24, 'h': 9}),
    ]
)

report.blocks = report.blocks[:1] + [wr.H1("Panel Grid Example"), pg] + report.blocks[1:]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[123] }} />

## Add data lineage with Artifact blocks {#add-data-lineage-with-artifact-blocks}

-   There are equivalent weave panels as well

``` python
artifact_lineage = wr.WeaveBlockArtifact(entity=ENTITY, project=LINEAGE_PROJECT, artifact='model-1', tab='lineage')

report.blocks = report.blocks[:1] + [wr.H1("Artifact lineage example"), artifact_lineage] + report.blocks[1:]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[124] }} />

## Customize run colors {#customize-run-colors}

-   Pass in a `dict[run_name, color]`

``` python
pg.custom_run_colors = {
  'adventurous-aardvark-1': '#e84118',
  'bountiful-badger-2':     '#fbc531',
  'clairvoyant-chipmunk-3': '#4cd137',
  'dastardly-duck-4':       '#00a8ff',
  'eloquent-elephant-5':    '#9c88ff',
}
report.save()
```

``` text
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[125] }} />

## Customize run sets with grouping, filtering, and ordering {#customize-run-sets-with-grouping-filtering-and-ordering}

-   Click on the different run sets in the iframe below and see how they are different
-   Grouping: Pass in a list of columns to group by
-   Filtering: Use `set_filters_with_python_expr` and pass in a valid python expression. The syntax is similar to `pandas.DataFrame.query`
-   Ordering: Pass in a list of columns where each value is prefixed with `+` for ascending or `-` for descending.

``` python
pg.runsets = [
    wr.Runset(ENTITY, PROJECT, name="Grouping", groupby=["encoder"]),
    wr.Runset(ENTITY, PROJECT, name="Filtering").set_filters_with_python_expr("encoder == 'resnet18' and loss < 0.05"),
    wr.Runset(ENTITY, PROJECT, name="Ordering", order=["+momentum", "-Name"]),
]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[126] }} />

## Customize group colors {#customize-group-colors}

-   Pass in a `dict[ordertuple, color]`, where `ordertuple: tuple[runset_name, *groupby_values]`
-   For example:
    -   Your runset is named `MyRunset`
    -   Your runset groupby is `["encoder", "optimizer"]`
    -   Then your tuple can be `("MyRunset", "resnet18", "adam")`

``` python
pg.custom_run_colors = {
  ('Grouping', 'resnet50'): 'red',
  ('Grouping', 'resnet18'): 'blue',
  
  # you can do both grouped and ungrouped colors in the same dict
  'adventurous-aardvark-1': '#e84118',
  'bountiful-badger-2':     '#fbc531',
  'clairvoyant-chipmunk-3': '#4cd137',
  'dastardly-duck-4':       '#00a8ff',
  'eloquent-elephant-5':    '#9c88ff',
}
report.save()
```

``` text
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
wandb: WARNING Multiple runs with the same name found! Using the first one.
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[127] }} />

# ❓ FAQ {#faq}

## My chart is not rendering as expected. {#my-chart-is-not-rendering-as-expected.}

-   We try to guess the column type, but sometimes we fail.
-   Try prefixing:
    -   `c::` for config values
    -   `s::` for summary metrics
    -   e.g. if your config value was `optimizer`, try `c::optimizer`

## My report is too wide/narrow {#my-report-is-too-widenarrow}

-   Change the report’s width to the right size for you.

``` python
report2 = report.save(clone=True)
report2.width = 'fluid'
report2.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[128] }} />

## How do I resize panels? {#how-do-i-resize-panels}

-   Pass a `dict[dim, int]` to `panel.layout`
-   `dim` is a dimension, which can be `x`, `y` (the coordiantes of the top left corner) `w`, `h` (the size of the panel)
-   You can pass any or all dimensions at once
-   The space between two dots in a panel grid is 2.

``` python
report = wr.Report(
    PROJECT,
    title="Resizing panels",
    description="Look at this wide parallel coordinates plot!",
    blocks=[
        wr.PanelGrid(
            panels=[
                wr.ParallelCoordinatesPlot(
                    columns=["Step", "c::model", "c::optimizer", "val_acc", "val_loss"],
                    layout={'w': 24, 'h': 9}  # change the layout!
                ),
            ]
        )
    ]
)
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[129] }} />

## What blocks are available? {#what-blocks-are-available}

-   See `wr.blocks` for a list of blocks.
-   In an IDE or notebook, you can also do `wr.blocks.<tab>` to get autocomplete.

``` python
# Panel grid is omitted.  See next section for PanelGrid and panels
report = wr.Report(
    PROJECT,
    title='W&B Block Gallery',
    description="Check out all of the blocks available in W&B",
    blocks=[
        wr.H1(text="Heading 1"),
        wr.P(text="Normal paragraph"),
        wr.H2(text="Heading 2"),
        wr.P(
            [
                "here is some text, followed by",
                wr.InlineCode("select * from code in line"),
                "and then latex",
                wr.InlineLaTeX("e=mc^2"),
            ]
        ),
        wr.H3(text="Heading 3"),
        wr.CodeBlock(
            code=["this:", "- is", "- a", "cool:", "- yaml", "- file"],
            language="yaml",
        ),
        wr.WeaveBlockSummaryTable(ENTITY, PROJECT, 'my-table'),
        wr.WeaveBlockArtifact(ENTITY, LINEAGE_PROJECT, 'model-1', 'lineage'),
        wr.WeaveBlockArtifactVersionedFile(ENTITY, LINEAGE_PROJECT, 'model-1', 'v0', "dataframe.table.json"),
        wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$"),
        wr.LaTeXBlock(text="\\gamma^2+\\theta^2=\\omega^2\n\\\\ a^2 + b^2 = c^2"),
        wr.Image("https://api.wandb.ai/files/megatruong/images/projects/918598/350382db.gif", caption="It's a me, Pikachu"),
        wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
        wr.OrderedList(items=["Ordered 1", "Ordered 2"]),
        wr.CheckedList(items=["Unchecked", "Checked"], checked=[False, True]),
        wr.BlockQuote(text="Block Quote 1\nBlock Quote 2\nBlock Quote 3"),
        wr.CalloutBlock(text=["Callout 1", "Callout 2", "Callout 3"]),
        wr.HorizontalRule(),
        wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
        wr.Spotify(spotify_id="5cfUlsdrdUE4dLMK7R9CFd"),
        wr.SoundCloud(url="https://api.soundcloud.com/tracks/1076901103"),
    ]
).save()
report.blocks += [wr.Gallery(ids=[report.id])]  # get report id on save
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[130] }} />

## What panels are available? {#what-panels-are-available}

-   See `wr.panels` for a list of panels
-   In an IDE or notebook, you can also do `wr.panels.<tab>` to get autocomplete.
-   Panels have a lot of settings. Inspect the panel to see what you can do!

``` python
report = wr.Report(
    project=PROJECT,
    title='W&B Panel Gallery',
    description="Check out all of the panels available in W&B",
    width='fluid',
    blocks=[
        wr.PanelGrid(
            runsets=[
                wr.Runset(project=LINEAGE_PROJECT),
                wr.Runset(),
            ],
            panels=[
                wr.MediaBrowser(media_keys="img"),
                wr.MarkdownPanel("Hello *italic* **bold** $e=mc^2$ `something`"),
                
                # LinePlot showed with many settings enabled for example
                wr.LinePlot(
                    title="Validation Accuracy over Time",
                    x="Step",
                    y=["val_acc"],
                    range_x=[0, 1000],
                    range_y=[1, 4],
                    log_x=True,
                    title_x="Training steps",
                    title_y="Validation Accuracy",
                    ignore_outliers=True,
                    groupby='encoder',
                    groupby_aggfunc="mean",
                    groupby_rangefunc="minmax",
                    smoothing_factor=0.5,
                    smoothing_type="gaussian",
                    smoothing_show_original=True,
                    max_runs_to_show=10,
                    font_size="large",
                    legend_position="west",
                ),
                wr.ScatterPlot(
                    title="Validation Accuracy vs. Validation Loss",
                    x="val_acc",
                    y="val_loss",
                    log_x=False,
                    log_y=False,
                    running_ymin=True,
                    running_ymean=True,
                    running_ymax=True,
                    font_size="small",
                    regression=True,
                ),
                wr.BarPlot(
                    title="Validation Loss by Encoder",
                    metrics=["val_loss"],
                    orientation='h',
                    range_x=[0, 0.11],
                    title_x="Validation Loss",
                    # title_y="y axis title",
                    groupby='encoder',
                    groupby_aggfunc="median",
                    groupby_rangefunc="stddev",
                    max_runs_to_show=20,
                    max_bars_to_show=3,
                    font_size="auto",
                ),
                wr.ScalarChart(
                    title="Maximum Number of Steps",
                    metric="Step",
                    groupby_aggfunc="max",
                    groupby_rangefunc="stderr",
                    font_size="large",
                ),
                wr.CodeComparer(diff="split"),
                wr.ParallelCoordinatesPlot(
                    columns=["Step", "c::model", "c::optimizer", "Step", "val_acc", "val_loss"],
                ),
                wr.ParameterImportancePlot(with_respect_to="val_loss"),
                wr.RunComparer(diff_only="split"),
                wr.CustomChart(
                    query={'summary': ['val_loss', 'val_acc']},
                    chart_name='wandb/scatter/v0',
                    user_fields={'x': 'val_loss', 'y': 'val_acc'}
                ),
                wr.WeavePanelSummaryTable("my-table"),
                wr.WeavePanelArtifact('model-1', 'lineage', layout={'w': 24, 'h': 12}),
                wr.WeavePanelArtifactVersionedFile('model-1', 'v0', "dataframe.table.json", layout={'w': 24, 'h': 12}),
            ],
        ),
    ]
)
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[131] }} />

## What above weave? {#what-above-weave}

-   A limited subset of weave panels are available today, including Artifact, ArtifactVersionedFile, and SummaryTable
-   Stay tuned for future updates around weave support!

## Can I use this in CI (e.g. create a report for each git commit?) {#can-i-use-this-in-ci-e.g.-create-a-report-for-each-git-commit}

-   Yep! Check out [this example](https://github.com/andrewtruong/wandb-gh-actions/actions/runs/3476558992) which creates a report via Github Actions

## How can I link related reports together? {#how-can-i-link-related-reports-together}

-   Suppose have have two reports like below:

``` python
report1 = wr.Report(
    PROJECT,
    title='Report 1',
    description="Great content coming from Report 1",
    blocks=[
        wr.H1('Heading from Report 1'),
        wr.P('Lorem ipsum dolor sit amet. Aut fuga minus nam vero saepeA aperiam eum omnis dolorum et ducimus tempore aut illum quis aut alias vero. Sed explicabo illum est eius quianon vitae sed voluptatem incidunt. Vel architecto assumenda Ad voluptatem quo dicta provident et velit officia. Aut galisum inventoreSed dolore a illum adipisci a aliquam quidem sit corporis quia cum magnam similique.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='bigfish', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'bigfish', 'ppg_procgen'): "#2980b9",
                ('Run set', 'bigfish', 'ppo_procgen'): "#e74c3c",
            }
        ),
    ]
).save()

report2 = wr.Report(
    PROJECT,
    title='Report 2',
    description="Great content coming from Report 2",
    blocks=[
        wr.H1('Heading from Report 2'),
        wr.P('Est quod ducimus ut distinctio corruptiid optio qui cupiditate quibusdam ea corporis modi. Eum architecto vero sed error dignissimosEa repudiandae a recusandae sint ut sint molestiae ea pariatur quae. In pariatur voluptas ad facere neque 33 suscipit et odit nostrum ut internos molestiae est modi enim. Et rerum inventoreAut internos et dolores delectus aut Quis sunt sed nostrum magnam ab dolores dicta.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/SPS']),
                wr.LinePlot(x='global_step', y=['charts/episodic_length']),
                wr.LinePlot(x='global_step', y=['charts/episodic_return']),
            ],
            runsets=[
                wr.Runset("openrlbenchmark", "cleanrl", "DQN", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'dqn_atari'"),
                wr.Runset("openrlbenchmark", "cleanrl", "SAC-discrete 0.8", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'sac_atari' and target_entropy_scale == 0.8"),
                wr.Runset("openrlbenchmark", "cleanrl", "SAC-discrete 0.88", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'sac_atari' and target_entropy_scale == 0.88"),
            ],
            custom_run_colors={
                ('DQN',               'dqn_atari'): '#e84118',
                ('SAC-discrete 0.8',  'sac_atari'): '#fbc531',
                ('SAC-discrete 0.88', 'sac_atari'): '#00a8ff',
            }
        ),
    ]
).save()
```

``` text
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
```

### Combine blocks into a new report {#combine-blocks-into-a-new-report}

``` python
report = wr.Report(PROJECT,
    title="Report with links",
    description="Use `wr.Link(text, url)` to add links inside normal text, or use normal markdown syntax in a MarkdownBlock",
    blocks=[
        wr.H1("This is a normal heading"),
        wr.P("And here is some normal text"),

        wr.H1(["This is a heading ", wr.Link("with a link!", url="https://wandb.ai/")]),
        wr.P(["Most text formats support ", wr.Link("adding links", url="https://wandb.ai/")]),

        wr.MarkdownBlock("""You can also use markdown syntax for [links](https://wandb.ai/)""")
    ]
)
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[132] }} />

``` python
report3 = wr.Report(
    PROJECT,
    title="Combined blocks report",
    description="This report combines blocks from both Report 1 and Report 2",
    blocks=[*report1.blocks, *report2.blocks]
)
report3.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[133] }} />

### Reference the two reports in a gallery block {#reference-the-two-reports-in-a-gallery-block}

``` python
report4 = wr.Report(
    PROJECT,
    title="Referenced reports via Gallery",
    description="This report has gallery links to Report1 and Report 2",
    blocks=[wr.Gallery(ids=[report1.id, report2.id])]
)
report4.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[134] }} />

## How do I add links to text? {#how-do-i-add-links-to-text}

-   Use `wr.Link`

## Do you support markdown? {#do-you-support-markdown}

Yep - In blocks, use `wr.MarkdownBlock` - In panels, use `wr.MarkdownPanel`

``` python
markdown = """
# Ducat quicquam

## Egit sit utque torta cuncta si oret

Lorem markdownum voco sacrorum religata. Erat ante condita mellaque Cimmerios
cultoribus pictas manu usquam illa umbra potentia inpedienda Padumque [euntes
motae](http://nec.com/necsollemnia.aspx), detraxit! Accessus discrimen,
Cyllenius *e* solum saepe terras perfringit amorem tenent consulit falce
referemus tantum. Illo qui attonitas, Adonis ultra stabunt horret requiescere
quae deam similis miserum consuetas: tantos aegram, metuam. Tetigere
**invidiae** preces indicere populo semper, limine sui dumque, lustra
alimentaque vidi nec corpusque aquarum habebat, in.


## Aurea simile iunctoque dux semper verbis

Vinctorum vidisset et caede officio, viae alia ratione aer regalia, etiamnum.
Occupat tanta; vicem, Ithaceque, et ille nec exclamat. Honori carpserat visae
magniloquo perluitur corpora tamen. Caput an Minervae sed vela est cecidere
luctus umbras iunctisque referat. Accensis **aderis capillos** pendebant
[retentas parvum](http://ipse.com/).

    if (desktop(2)) {
        laser_qwerty_optical.webcam += upsRoom + window;
    }
    if (type) {
        memoryGrayscale(backbone, mask_multimedia_html);
    }
    if (natInterfaceFile == 23 + 92) {
        interface_sku.platform = compressionMotherboard - error_veronica_ata;
        dsl_windows = 57 * -2;
        definition *= -4;
    } else {
        frame(4, market_chip_irq, megapixel_eide);
    }
    if (mashupApiFlash(-1, margin) - graphicSoftwareNas.ddr_samba_reimage(port)
            != control_navigation_pseudocode(yahoo.microcomputerDimm(
            mips_adsl))) {
        postscriptViralDirectx(1, cron_router_voip(669103, managementPitch,
                ospf_up_paper), frame);
        servlet_cross_paper.controlLanguage(insertion_source.viewHorizontalRead(
                enterprise, widget, parse_encoding), end);
        script_e(rateRss(yobibyte, fddi, vci_hyper_joystick), surgeHeat / case);
    }
    pcmciaRealSystem.basic_exbibyte_controller = carrier.domainDesktop(-4 +
            laptop + 5);
"""

report = wr.Report(
    PROJECT,
    title="Report with markdown",
    description="See what's possible with MarkdownBlock and MarkdownPanel",
    blocks=[
        wr.MarkdownBlock(markdown),
        wr.PanelGrid(
            panels=[
                wr.MarkdownPanel(markdown,                layout={'w': 12, 'h': 18}),
                wr.LinePlot(x='Step', y=['val_acc'] ,     layout={'x': 12, 'y': 0}),
                wr.LinePlot(x='Step', y=['val_loss'],     layout={'x': 12, 'y': 6}),
                wr.ScatterPlot(x='val_loss', y='val_acc', layout={'x': 12, 'y': 12}),
            ]
        )
    ]
)
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[135] }} />

## Can I build the report up from smaller pieces / all at once? {#can-i-build-the-report-up-from-smaller-pieces-all-at-once}

Yep. We’ll demonstrate by putting together a report with a parallel coordinates plot.

NOTE: this section assumes you have run the [sweeps notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) already.

### Build it up incrementally {#build-it-up-incrementally}

As you might do if you were creating a report in the UI

1.  Create a report

``` python
report = wr.Report(project=PROJECT, title='Parallel Coordinates Example', description="Using the pytorch sweeps demo")
```

1.  Add a panel grid

``` python
pg = wr.PanelGrid()
report.blocks = [pg]
```

1.  Specify your runsets

``` python
pg.runsets = [wr.Runset(project='pytorch-sweeps-demo')]
```

1.  Specify your panels

``` python
pg.panels = [
    wr.ParallelCoordinatesPlot(
        columns=["c::batch_size", "c::dropout", "c::epochs", "c::fc_layer_size", "c::learning_rate", "c::optimizer", "loss"]
    )
]
```

1.  Save the report

``` python
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[136] }} />

### The same thing all-in-one {#the-same-thing-all-in-one}

``` python
report = wr.Report(
    project=PROJECT,
    title="Parallel Coordinates Example (all-in-one)",
    description="Using the pytorch sweeps demo (same as the other one but written in one expression)",
    blocks=[
        wr.PanelGrid(
            runsets=[wr.Runset(project="pytorch-sweeps-demo")],
            panels=[
                wr.ParallelCoordinatesPlot(
                    columns=['c::batch_size', 'c::dropout', 'c::epochs', 'c::fc_layer_size', 'c::learning_rate', 'c::optimizer', 'loss']
                )
            ],
        )
    ],
)
```

``` python
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[137] }} />

## I tried mutating an object in list but it didn’t work! {#i-tried-mutating-an-object-in-list-but-it-didnt-work}

tl;dr: It should always work if you assign a value to the attribute instead of mutating. If you really need to mutate, do it before assignment.

------------------------------------------------------------------------

This can happen in a few places that contain lists of wandb objects, e.g.: - `report.blocks` - `panel_grid.panels` - `panel_grid.runsets`

``` python
report = wr.Report(project=PROJECT)
```

Good: Assign `b`

``` python
b = wr.H1(text=["Hello", " World!"])
report.blocks = [b]
assert b.text == ["Hello", " World!"]
assert report.blocks[0].text == ["Hello", " World!"]
```

Bad: Mutate `b` without reassigning

``` python
b.text = ["Something", " New"]
assert b.text == ["Something", " New"]
assert report.blocks[0].text == ["Hello", " World!"]
```

Good: Mutate `b` and then reassign it

``` python
report.blocks = [b]
assert b.text == ["Something", " New"]
assert report.blocks[0].text == ["Something", " New"]
```

## How do I show tables? {#how-do-i-show-tables}

``` python
report = wr.Report(project=PROJECT, title='Adding tables to reports', description="Add tables with WeaveBlockSummaryTable or WeavePanelSummaryTable")
```

### Using weave blocks {#using-weave-blocks}

``` python
report.blocks += [wr.WeaveBlockSummaryTable(ENTITY, PROJECT, "my-table")]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[138] }} />

### Using weave panels (via PanelGrid) {#using-weave-panels-via-panelgrid}

``` python
report.blocks += [
    wr.PanelGrid(
        panels=[wr.WeavePanelSummaryTable("my-table")]
    )
]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[139] }} />

## How do I show artifact lineage / versions? {#how-do-i-show-artifact-lineage-versions}

``` python
report = wr.Report(project=PROJECT, title='Adding artifact lineage to reports', description="via WeaveBlockArtifact, WeaveBlockArtifactVersionedFile, or their panel equivalents")
```

### Using weave blocks {#using-weave-blocks-1}

``` python
report.blocks += [
    wr.WeaveBlockArtifact(ENTITY, LINEAGE_PROJECT, "model-1", "lineage"),
    wr.WeaveBlockArtifactVersionedFile(ENTITY, LINEAGE_PROJECT, "model-1", "v0", "dataframe.table.json")
]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[140] }} />

### Using weave panels (via PanelGrid) {#using-weave-panels-via-panelgrid-1}

``` python
report.blocks += [
    wr.PanelGrid(panels=[
        wr.WeavePanelArtifact("model-1", "lineage"),
        wr.WeavePanelArtifactVersionedFile("model-1", "v0", "dataframe.table.json")
    ])
]
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[141] }} />

## How can I create report templates? {#how-can-i-create-report-templates}

-   See some of the examples in `wr.templates`
-   The most straightforward way is to create a function that returns your target report and/or its blocks.

### A basic template {#a-basic-template}

-   Just use a function

``` python
def my_report_template(title, description, project, metric):
    return wr.Report(
        title=title,
        description=description,
        project=project,
        blocks=[
            wr.H1(f"Look at our amazing metric called `{metric}`"),
            wr.PanelGrid(
                panels=[wr.LinePlot(x='Step', y=metric, layout={'w': 24, 'h': 8})],
            )
        ]
    ).save()
```

``` python
my_report_template('My templated report', "Here's an example of how you can make a function for templates", PROJECT, 'val_acc')
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[142] }} />

### More advanced templates {#more-advanced-templates}

``` python
def create_header():
  return [
    wr.P(),
    wr.HorizontalRule(),
    wr.P(),
    wr.Image(
      "https://camo.githubusercontent.com/83839f20c90facc062330f8fee5a7ab910fdd04b80b4c4c7e89d6d8137543540/68747470733a2f2f692e696d6775722e636f6d2f676236423469672e706e67"
    ),
    wr.P(),
    wr.HorizontalRule(),
    wr.P(),
  ]

def create_footer():
  return [
    wr.P(),
    wr.HorizontalRule(),
    wr.P(),
    wr.H1("Disclaimer"),
    wr.P(
      "The views and opinions expressed in this report are those of the authors and do not necessarily reflect the official policy or position of Weights & Biases. blah blah blah blah blah boring text at the bottom"
    ),
    wr.P(),
    wr.HorizontalRule(),
  ]

def create_main_content(metric):
  return [
    wr.H1(f"Look at our amazing metric called `{metric}`"),
    wr.PanelGrid(
      panels=[wr.LinePlot(x='Step', y=metric, layout={'w': 24, 'h': 8})],
    )
  ]

def create_templated_report_with_header_and_footer(title, project, metric):
  return wr.Report(
    title=title,
    project=project,
    blocks=[
      *create_header(),
      *create_main_content(metric),
      *create_footer(),
    ]).save()
```

``` python
create_templated_report_with_header_and_footer(title="Another templated report", project=PROJECT, metric='val_acc')
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[143] }} />

# 🧩 Patterns {#patterns}

# 📌 Complete Examples {#complete-examples}

## Reinforcement Learning (RL) {#reinforcement-learning-rl}

``` python
wr.Report(
    project=PROJECT,
    title='Reinforcement Learning Report',
    description='Aut totam dolores aut galisum atque aut placeat quia. Vel quisquam omnis ut quibusdam doloremque a delectus quia in omnis deserunt. Quo ipsum beatae aut veniam earum non ipsa reiciendis et fugiat asperiores est veritatis magni et corrupti internos. Ut quis libero ut alias reiciendis et animi delectus.',
    blocks=[
        wr.TableOfContents(),
        wr.H1("Ea quidem illo est dolorem illo."),
        wr.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ac eros ut nunc venenatis tincidunt vel ut dolor. Sed sed felis dictum, congue risus vel, aliquet dolor. Donec ut risus vel leo dictum tristique. Nunc sed urna mi. Morbi nulla turpis, vehicula eu maximus ut, gravida id libero. Duis porta risus leo, quis lobortis enim ultrices a. Donec quam augue, vestibulum vitae mollis at, tincidunt non orci. Morbi faucibus dignissim tempor. Vestibulum ornare augue a orci tincidunt porta. Pellentesque et ante et purus gravida euismod. Maecenas sit amet sollicitudin felis, sed egestas nunc."),
        wr.H2('Et sunt sunt eum asperiores ratione.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='bigfish', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'bigfish', 'ppg_procgen'): "#2980b9",
                ('Run set', 'bigfish', 'ppo_procgen'): "#e74c3c",
            }
        ),
        wr.H2('Sit officia inventore non omnis deleniti.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='starpilot', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'starpilot', 'ppg_procgen'): "#2980b9",
                ('Run set', 'starpilot', 'ppo_procgen'): "#e74c3c",
            }
        ),
        wr.H2('Aut amet nesciunt vel quisquam repellendus sed labore voluptas.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='bossfight', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'bossfight', 'ppg_procgen'): "#2980b9",
                ('Run set', 'bossfight', 'ppo_procgen'): "#e74c3c",
            }
        ),
        wr.HorizontalRule(),
        wr.H1("Sed consectetur vero et voluptas voluptatem et adipisci blanditiis."),
        wr.P("Sit aliquid repellendus et numquam provident quo quaerat earum 33 sunt illo et quos voluptate est officia deleniti. Vel architecto nulla ex nulla voluptatibus qui saepe officiis quo illo excepturi ea dolorum reprehenderit."),
        wr.H2("Qui debitis iure 33 voluptatum eligendi."),
        wr.P("Non veniam laudantium et fugit distinctio qui aliquid eius sed laudantium consequatur et quia perspiciatis. Et odio inventore est voluptas fugiat id perspiciatis dolorum et perferendis recusandae vel Quis odio 33 beatae veritatis. Ex sunt accusamus aut soluta eligendi sed perspiciatis maxime 33 dolorem dolorum est aperiam minima. Et earum rerum eos illo sint eos temporibus similique ea fuga iste sed quia soluta sit doloribus corporis sed tenetur excepturi?"),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/SPS']),
                wr.LinePlot(x='global_step', y=['charts/episodic_length']),
                wr.LinePlot(x='global_step', y=['charts/episodic_return']),
            ],
            runsets=[
                wr.Runset("openrlbenchmark", "cleanrl", "DQN", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'dqn_atari'"),
                wr.Runset("openrlbenchmark", "cleanrl", "SAC-discrete 0.8", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'sac_atari' and target_entropy_scale == 0.8"),
                wr.Runset("openrlbenchmark", "cleanrl", "SAC-discrete 0.88", groupby=["exp_name"]).set_filters_with_python_expr("env_id == 'BreakoutNoFrameskip-v4' and exp_name == 'sac_atari' and target_entropy_scale == 0.88"),
            ],
            custom_run_colors={
                ('DQN',               'dqn_atari'): '#e84118',
                ('SAC-discrete 0.8',  'sac_atari'): '#fbc531',
                ('SAC-discrete 0.88', 'sac_atari'): '#00a8ff',
            }
        ),
    ]
).save()
```

``` text
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[144] }} />

## Customer Landing Page {#customer-landing-page}

``` python
report = wr.templates.create_customer_landing_page(
    project=PROJECT,
    company_name='Company',
    main_contact='Contact McContact (email@company.com)',
    slack_link='https://company.slack.com/blah',
)
report.save()
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[145] }} />

## Enterprise Report with Branded Header and Footer {#enterprise-report-with-branded-header-and-footer}

``` python
report = wr.templates.create_enterprise_report(
    project=PROJECT,
    body=[
        wr.H1("Ea quidem illo est dolorem illo."),
        wr.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ac eros ut nunc venenatis tincidunt vel ut dolor. Sed sed felis dictum, congue risus vel, aliquet dolor. Donec ut risus vel leo dictum tristique. Nunc sed urna mi. Morbi nulla turpis, vehicula eu maximus ut, gravida id libero. Duis porta risus leo, quis lobortis enim ultrices a. Donec quam augue, vestibulum vitae mollis at, tincidunt non orci. Morbi faucibus dignissim tempor. Vestibulum ornare augue a orci tincidunt porta. Pellentesque et ante et purus gravida euismod. Maecenas sit amet sollicitudin felis, sed egestas nunc."),
        wr.H2('Et sunt sunt eum asperiores ratione.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='bigfish', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'bigfish', 'ppg_procgen'): "#2980b9",
                ('Run set', 'bigfish', 'ppo_procgen'): "#e74c3c",
            }
        ),
        wr.H2('Sit officia inventore non omnis deleniti.'),
        wr.PanelGrid(
            panels=[
                wr.LinePlot(x='global_step', y=['charts/episodic_return'], smoothing_factor=0.85, groupby_aggfunc='mean', groupby_rangefunc='minmax', layout={'x': 0, 'y': 0, 'w': 12, 'h': 8}),
                wr.MediaBrowser(media_keys="videos", num_columns=4, layout={'w': 12, 'h': 8}),
            ],
            runsets=[
                wr.Runset(entity='openrlbenchmark', project='cleanrl', query='starpilot', groupby=['env_id', 'exp_name'])
            ],
            custom_run_colors={
                ('Run set', 'starpilot', 'ppg_procgen'): "#2980b9",
                ('Run set', 'starpilot', 'ppo_procgen'): "#e74c3c",
            }
        ),
    ]
)
report.save()
```

``` text
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
wandb: WARNING A graphql request initiated by the public wandb API timed out (timeout=9 sec). Create a new API with an integer timeout larger than 9, e.g., `api = wandb.Api(timeout=19)` to increase the graphql timeout.
```

<div dangerouslySetInnerHTML={{ __html: quartoRawHtml[146] }} />
