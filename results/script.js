const jsonDir = "json-data"

async function loadAllData() {
    const indexResponse = await fetch(jsonDir + "/index.json")
    const files = await indexResponse.json()

    const datasets = await Promise.all(files.map(async (file) => {
        const response = await fetch(`${jsonDir}/${file}`)
        return response.json()
    }))

    return datasets
}

function strToPre(str) {
    pre = document.createElement("pre")
    code = document.createElement("code")
    code.classList.add("language-python")
    code.textContent = database[0]["fn"]
    pre.appendChild(code)
    hljs.highlightElement(code)
    return pre
}

function improvementCanvas(islands) {
    const timelabels = [...Array(Math.max(...islands.map(island => island.runs.length))).keys()]
    const improvementsCanvas = document.createElement("canvas")
    new Chart(
        improvementsCanvas,
        {
            type: 'line',
            data: {
                labels: timelabels,
                datasets: islands.map(island => ({
                    label: `Island ${island.ix}`,
                    data: island.runningMaximum,
                    borderColor: colors[island.ix % colors.length],
                    pointRadius: 0,
                })
                )
            },
            options: {
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: "Run Index"
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: "Running Max Score"
                        }
                    }
                }
            }
        }
    )
    return improvementsCanvas
}

function getProblemContainer(problemName) {
    const maybeExisting = document.getElementById(`container-${problemName}`)
    if (maybeExisting) {
        return maybeExisting
    }

    const div = document.createElement("div")
    div.id = `container-${problemName}`
    div.classList.add("problem-container")
    document.body.appendChild(div)

    const heading = document.createElement("h2")
    heading.textContent = problemName
    div.appendChild(heading)

    return div
}

function details(title, description, ...content) {
    const details = document.createElement("details")
    const summary = document.createElement("summary")
    details.appendChild(summary)

    const containerInner = document.createElement("div")
    containerInner.classList.add("details-inner")
    details.appendChild(containerInner)

    const titleSpan = document.createElement("span")
    titleSpan.textContent = title
    titleSpan.classList.add("title")
    summary.appendChild(titleSpan)

    if (description) {
        const descriptionSpan = document.createElement("span")
        descriptionSpan.textContent = description
        descriptionSpan.classList.add("description")
        summary.appendChild(descriptionSpan)
    }

    content.forEach(elem => containerInner.appendChild(elem))
    return details
}

function detailsCode(title, description, code) {
    const pre = document.createElement("pre")
    const codeElement = document.createElement("code")
    codeElement.classList.add("language-python")
    codeElement.textContent = code
    pre.appendChild(codeElement)
    hljs.highlightElement(codeElement)
    return details(title, description, pre)
}

// Paul Tol's discrete rainbow color scheme, from https://personal.sron.nl/~pault/
const colors = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499', '#888']

async function displayDatabase(database) {
    /* Schema from generate.py (might be outdated)
      {
        "problemName": database.problem_name,
        "inputs": database.inputs,
        "message": database.message,
        "config": vars(database._config),  # noqa: SLF001
        "specCode": database._specification,  # noqa: SLF001
        "timestamp": database.timestamp,
        "islands": [
          {
            "improvements": [(ix, island._runs[ix], str(program)) for ix, program in island._improvements],  # noqa: SLF001
            "successCount": island._success_count,  # noqa: SLF001
            "failureCount": island._failure_count,  # noqa: SLF001
          }
          for island in database._islands  # noqa: SLF001
        ],
      },
    */
    const problemName = database.problemName
    const problemContainer = getProblemContainer(problemName)
    const islands = database.islands.map((island, i) => {
        const lastImprovement = island.improvements[island.improvements.length - 1]
        island.bestScore = lastImprovement[1]
        island.ix = i

        return island
    })
    islands.sort((a, b) => b.bestScore - a.bestScore)
    const maxScore = islands[0].bestScore
    const totalSuccesses = islands.reduce((acc, island) => acc + island.successCount, 0)
    const totalFailures = islands.reduce((acc, island) => acc + island.failureCount, 0)
    const totalRate = Math.round(100 * totalSuccesses / (totalSuccesses + totalFailures))


    const runDetails = details(`${problemName}(${database.inputs.join(', ')}) → ${maxScore}`, database.message,
        detailsCode("Spec", "Specification-file for this problem and run", database.specCode),
        details("Best Programs", "Best program of each island", ...islands.map(island => detailsCode(`Score ${island.bestScore}`, `Island ${island.ix}`, island.improvements[island.improvements.length - 1][2])
        )),
        details("Improvements over Time", "Improvement-steps of each island", improvementCanvas(islands),
            ...islands.map(island => details(`Score ${island.bestScore}`, `Island ${island.ix}`,
                ...island.improvements.toReversed().map(improvement => detailsCode(`Score ${improvement[1]}`, `Run ${improvement[0]}`, improvement[2])))
            )),
        details(`Error-rates`, `Total ${totalRate}%`, errorCanvas(islands)),
        detailsCode("Config", "Config-file for this run", Object.entries(database.config).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n")),
    )
    problemContainer.appendChild(runDetails)

    // Let's just hope these are unique.
    runDetails.id = `run-${database.timestamp}`
    runDetails.classList.add("run-container")

    const messageSpan = document.createElement("span")
    messageSpan.textContent = database.message
    messageSpan.classList.add("pre-wrap")
    const runDetailsInner = runDetails.querySelector(".details-inner")
    runDetailsInner.insertBefore(messageSpan.cloneNode(true), runDetailsInner.firstChild)

    const timestampLink = document.createElement("a")
    timestampLink.classList.add("timestamp")
    const href = `#run-${problemName}-${database.timestamp}`
    timestampLink.href = href
    timestampLink.textContent = href
    runDetails.querySelector("summary").appendChild(timestampLink)
}

async function main() {
    const databases = await loadAllData()
    databases.forEach(displayDatabase)
}

hljs.addPlugin(new CopyButtonPlugin())

window.addEventListener('load', () => {
    main().then(() => {
        const hash = window.location.hash.slice(1)
        if (hash) {
            const element = document.getElementById(hash)
            if (element) {
                element.open = true
                element.scrollIntoView()
            }
        }
    })
})