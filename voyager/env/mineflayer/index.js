const fs = require("fs");
const express = require("express");
const bodyParser = require("body-parser");
const mineflayer = require("mineflayer");

"TODO: image capture and metadata collection, openai api call"; 
const puppeteer = require('puppeteer');
//const { createCanvas, Image } = require("canvas");
const captureInterval = 60000; // 1 minute
const { mineflayer: mineflayerViewer } = require('prismarine-viewer');
const OpenAI = require('openai');
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const skills = require("./lib/skillLoader");
const { initCounter, getNextTime } = require("./lib/utils");
const obs = require("./lib/observation/base");
const OnChat = require("./lib/observation/onChat");
const OnError = require("./lib/observation/onError");
const { Voxels, BlockRecords } = require("./lib/observation/voxels");
const Status = require("./lib/observation/status");
const Inventory = require("./lib/observation/inventory");
const OnSave = require("./lib/observation/onSave");
const Chests = require("./lib/observation/chests");
const { plugin: tool } = require("mineflayer-tool");

"add debug mode";
const debug = require('debug')('bot');
"add error handling";

let bot = null;

const app = express();

app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ limit: "50mb", extended: false }));

// Add bot management
const bots = new Map();  // Store multiple bots
let currentBot = null; // Keep track of current bot for compatibility

app.post("/start", async (req, res) => {
    try {
        const serverPort = parseInt(req.app.get('port') || req.query.port || '3000');
        const mcPort = parseInt(req.body.port);
        const username = req.body.username;

        console.log(`Starting bot ${username} on server port ${serverPort}, MC port ${mcPort}`);

        // Clean up existing bot on this server port if any
        if (bots.has(serverPort)) {
            const existingBot = bots.get(serverPort);
            try {
                if (existingBot.viewer) existingBot.viewer.close();
                await existingBot.end();
            } catch (err) {
                console.error(`Error cleaning up existing bot on port ${serverPort}:`, err);
            }
            bots.delete(serverPort);
        }

        // Create new bot
        const bot = mineflayer.createBot({
            host: "localhost",
            port: mcPort,
            username: username,
            version: "1.19.4",
            viewDistance: "tiny",
            disableChatSigning: true
        });

        // Set as current bot for compatibility
        currentBot = bot;
        
        // Store in bots map
        bots.set(serverPort, bot);

        // Keep all existing plugin loading
        bot.loadPlugin(require('mineflayer-pathfinder').pathfinder);
        bot.loadPlugin(require('mineflayer-tool').plugin);
        bot.loadPlugin(require('mineflayer-collectblock').plugin);
        bot.loadPlugin(require('mineflayer-pvp').plugin);

        // Set up event handlers
        bot.once('spawn', () => {
            console.log(`Bot ${username} spawned successfully`);
            
            try {
                // Initialize viewer with unique port
                const viewerPort = 3007 + (serverPort - 3000);
                mineflayerViewer(bot, { port: viewerPort, firstPerson: true });
                console.log(`Viewer started on port ${viewerPort}`);

                // Keep existing game rule settings
                bot.chat("/gamerule doDaylightCycle false");
                bot.chat("/time set day");
                bot.chat("/gamerule doWeatherCycle false");
                bot.chat("/weather clear");
                bot.chat("/gamerule doMobSpawning false");
                bot.chat("/kill @e[type=!player]");

                res.json({ status: 'success' });
            } catch (err) {
                console.error(`Error in spawn handler:`, err);
                if (!res.headersSent) {
                    res.status(500).json({ error: err.message });
                }
            }
        });

        bot.on('error', (err) => {
            console.error(`Bot error:`, err);
            if (!res.headersSent) {
                res.status(500).json({ error: err.message });
            }
        });

        bot.on('end', () => {
            console.log(`Bot ${username} disconnected`);
            bots.delete(serverPort);
            if (currentBot === bot) {
                currentBot = null;
            }
        });

        // Add timeout for connection
        setTimeout(() => {
            if (!res.headersSent) {
                res.status(500).json({ error: 'Connection timeout' });
            }
        }, 30000);

    } catch (err) {
        console.error('Error in /start:', err);
        if (!res.headersSent) {
            res.status(500).json({ error: err.message });
        }
    }
});

app.post("/step", async (req, res) => {
    // import useful package
    let response_sent = false;
    function otherError(err) {
        console.log("Uncaught Error");
        bot.emit("error", handleError(err));
        bot.waitForTicks(bot.waitTicks).then(() => {
            if (!response_sent) {
                response_sent = true;
                res.json(bot.observe());
            }
        });
    }

    process.on("uncaughtException", otherError);

    const mcData = require("minecraft-data")(bot.version);
    mcData.itemsByName["leather_cap"] = mcData.itemsByName["leather_helmet"];
    mcData.itemsByName["leather_tunic"] =
        mcData.itemsByName["leather_chestplate"];
    mcData.itemsByName["leather_pants"] =
        mcData.itemsByName["leather_leggings"];
    mcData.itemsByName["leather_boots"] = mcData.itemsByName["leather_boots"];
    mcData.itemsByName["lapis_lazuli_ore"] = mcData.itemsByName["lapis_ore"];
    mcData.blocksByName["lapis_lazuli_ore"] = mcData.blocksByName["lapis_ore"];
    const {
        Movements,
        goals: {
            Goal,
            GoalBlock,
            GoalNear,
            GoalXZ,
            GoalNearXZ,
            GoalY,
            GoalGetToBlock,
            GoalLookAtBlock,
            GoalBreakBlock,
            GoalCompositeAny,
            GoalCompositeAll,
            GoalInvert,
            GoalFollow,
            GoalPlaceBlock,
        },
        pathfinder,
        Move,
        ComputedPath,
        PartiallyComputedPath,
        XZCoordinates,
        XYZCoordinates,
        SafeBlock,
        GoalPlaceBlockOptions,
    } = require("mineflayer-pathfinder");
    const { Vec3 } = require("vec3");

    // Set up pathfinder
    const movements = new Movements(bot, mcData);
    bot.pathfinder.setMovements(movements);

    bot.globalTickCounter = 0;
    bot.stuckTickCounter = 0;
    bot.stuckPosList = [];

    function onTick() {
        bot.globalTickCounter++;
        if (bot.pathfinder.isMoving()) {
            bot.stuckTickCounter++;
            if (bot.stuckTickCounter >= 100) {
                onStuck(1.5);
                bot.stuckTickCounter = 0;
            }
        }
    }

    bot.on("physicTick", onTick);

    // initialize fail count
    let _craftItemFailCount = 0;
    let _killMobFailCount = 0;
    let _mineBlockFailCount = 0;
    let _placeItemFailCount = 0;
    let _smeltItemFailCount = 0;

    // Retrieve array form post bod
    const code = req.body.code;
    const programs = req.body.programs;
    bot.cumulativeObs = [];
    await bot.waitForTicks(bot.waitTicks);
    const r = await evaluateCode(code, programs);
    process.off("uncaughtException", otherError);
    if (r !== "success") {
        bot.emit("error", handleError(r));
    }
    await returnItems();
    // wait for last message
    await bot.waitForTicks(bot.waitTicks);
    if (!response_sent) {
        response_sent = true;
        res.json(bot.observe());
    }
    bot.removeListener("physicTick", onTick);

    async function evaluateCode(code, programs) {
        // Echo the code produced for players to see it. Don't echo when the bot code is already producing dialog or it will double echo
        try {
            await eval("(async () => {" + programs + "\n" + code + "})()");
            return "success";
        } catch (err) {
            return err;
        }
    }

    function onStuck(posThreshold) {
        const currentPos = bot.entity.position;
        bot.stuckPosList.push(currentPos);

        // Check if the list is full
        if (bot.stuckPosList.length === 5) {
            const oldestPos = bot.stuckPosList[0];
            const posDifference = currentPos.distanceTo(oldestPos);

            if (posDifference < posThreshold) {
                teleportBot(); // execute the function
            }

            // Remove the oldest time from the list
            bot.stuckPosList.shift();
        }
    }

    function teleportBot() {
        const blocks = bot.findBlocks({
            matching: (block) => {
                return block.type === 0;
            },
            maxDistance: 1,
            count: 27,
        });

        if (blocks) {
            // console.log(blocks.length);
            const randomIndex = Math.floor(Math.random() * blocks.length);
            const block = blocks[randomIndex];
            bot.chat(`/tp @s ${block.x} ${block.y} ${block.z}`);
        } else {
            bot.chat("/tp @s ~ ~1.25 ~");
        }
    }

    function returnItems() {
        bot.chat("/gamerule doTileDrops false");
        const crafting_table = bot.findBlock({
            matching: mcData.blocksByName.crafting_table.id,
            maxDistance: 128,
        });
        if (crafting_table) {
            bot.chat(
                `/setblock ${crafting_table.position.x} ${crafting_table.position.y} ${crafting_table.position.z} air destroy`
            );
            bot.chat("/give @s crafting_table");
        }
        const furnace = bot.findBlock({
            matching: mcData.blocksByName.furnace.id,
            maxDistance: 128,
        });
        if (furnace) {
            bot.chat(
                `/setblock ${furnace.position.x} ${furnace.position.y} ${furnace.position.z} air destroy`
            );
            bot.chat("/give @s furnace");
        }
        if (bot.inventoryUsed() >= 32) {
            // if chest is not in bot's inventory
            if (!bot.inventory.items().find((item) => item.name === "chest")) {
                bot.chat("/give @s chest");
            }
        }
        // if iron_pickaxe not in bot's inventory and bot.iron_pickaxe
        if (
            bot.iron_pickaxe &&
            !bot.inventory.items().find((item) => item.name === "iron_pickaxe")
        ) {
            bot.chat("/give @s iron_pickaxe");
        }
        bot.chat("/gamerule doTileDrops true");
    }

    function handleError(err) {
        let stack = err.stack;
        if (!stack) {
            return err;
        }
        console.log(stack);
        const final_line = stack.split("\n")[1];
        const regex = /<anonymous>:(\d+):\d+\)/;

        const programs_length = programs.split("\n").length;
        let match_line = null;
        for (const line of stack.split("\n")) {
            const match = regex.exec(line);
            if (match) {
                const line_num = parseInt(match[1]);
                if (line_num >= programs_length) {
                    match_line = line_num - programs_length;
                    break;
                }
            }
        }
        if (!match_line) {
            return err.message;
        }
        let f_line = final_line.match(
            /\((?<file>.*):(?<line>\d+):(?<pos>\d+)\)/
        );
        if (f_line && f_line.groups && fs.existsSync(f_line.groups.file)) {
            const { file, line, pos } = f_line.groups;
            const f = fs.readFileSync(file, "utf8").split("\n");
            // let filename = file.match(/(?<=node_modules\\)(.*)/)[1];
            let source = file + `:${line}\n${f[line - 1].trim()}\n `;

            const code_source =
                "at " +
                code.split("\n")[match_line - 1].trim() +
                " in your code";
            return source + err.message + "\n" + code_source;
        } else if (
            f_line &&
            f_line.groups &&
            f_line.groups.file.includes("<anonymous>")
        ) {
            const { file, line, pos } = f_line.groups;
            let source =
                "Your code" +
                `:${match_line}\n${code.split("\n")[match_line - 1].trim()}\n `;
            let code_source = "";
            if (line < programs_length) {
                source =
                    "In your program code: " +
                    programs.split("\n")[line - 1].trim() +
                    "\n";
                code_source = `at line ${match_line}:${code
                    .split("\n")
                    [match_line - 1].trim()} in your code`;
            }
            return source + err.message + "\n" + code_source;
        }
        return err.message;
    }
});

app.post("/stop", (req, res) => {
    bot.end();
    res.json({
        message: "Bot stopped",
    });
});

app.post("/pause", (req, res) => {
    if (!bot) {
        res.status(400).json({ error: "Bot not spawned" });
        return;
    }
    bot.chat("/pause");
    bot.waitForTicks(bot.waitTicks).then(() => {
        res.json({ message: "Success" });
    });
});

// Server listening to PORT 3000

const DEFAULT_PORT = 3000;
const PORT = process.argv[2] || DEFAULT_PORT;
app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});

async function setupVisionCapture(bot) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.setViewport({ width: 640, height: 480 });
    await page.goto('http://localhost:3007'); // Connect to the viewer
    
    // Wait for the viewer to load
    await page.waitForTimeout(2000);

    let lastCaptureTime = Date.now();
    const captureInterval = 60000; // 1 minute

    async function captureAndAnalyze() {
        try {
            // Take screenshot
            const screenshot = await page.screenshot({
                encoding: 'base64'
            });

            // Collect metadata
            const metadata = {
                timestamp: new Date().toISOString(),
                position: bot.entity.position,
                orientation: bot.entity.yaw,
                inventory: bot.inventory.items().map(item => ({
                    name: item.name,
                    count: item.count
                }))
            };

            // Send to GPT-4 Vision
            const response = await openai.chat.completions.create({
                model: "gpt-4-vision-preview",
                messages: [
                    {
                        role: "system",
                        content: "You are analyzing Minecraft environments. Focus on identifying: 1) Resources worth collecting 2) Potential hazards 3) Navigation suggestions. Be concise."
                    },
                    {
                        role: "user",
                        content: [
                            {
                                type: "text",
                                text: `Analyze this Minecraft view and provide actionable insights.\nBot metadata: ${JSON.stringify(metadata, null, 2)}`
                            },
                            {
                                type: "image_url",
                                image_url: `data:image/png;base64,${screenshot}`
                            }
                        ]
                    }
                ],
                max_tokens: 300
            });

            const analysis = response.choices[0].message.content;
            console.log('Environment Analysis:', analysis);
            
            // Optionally, you can emit an event with the analysis
            bot.emit('environmentAnalysis', analysis);
            
            return analysis;

        } catch (error) {
            console.error('Vision analysis error:', error);
        }
    }

    // Set up periodic capture
    bot.on('move', async () => {
        const now = Date.now();
        if (now - lastCaptureTime > captureInterval) {
            await captureAndAnalyze();
            lastCaptureTime = now;
        }
    });

    // Clean up function
    const cleanup = async () => {
        await browser.close();
    };

    // Add cleanup to bot's end event
    bot.once('end', cleanup);

    return cleanup; // Return cleanup function in case you need to stop early
}

// Add server error handling
app.use((err, req, res, next) => {
    console.error("Express error:", err);
    if (!res.headersSent) {
        res.status(500).json({ error: err.message });
    }
});

// Graceful shutdown
process.on("SIGINT", async () => {
    console.log("Shutting down gracefully...");
    if (bot) {
        try {
            if (bot.viewer) {
                bot.viewer.close();
            }
            await bot.end();
        } catch (err) {
            console.error("Error during shutdown:", err);
        }
    }
    process.exit(0);
});
