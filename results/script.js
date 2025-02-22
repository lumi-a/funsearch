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

function errorCanvas(islands) {
    const errorCanvas = document.createElement("canvas")
    new Chart(
        errorCanvas,
        {
            type: 'scatter',
            data: {
                datasets: islands.map(island => ({
                    label: `Island ${island.ix}`,
                    data: island.runs.map((run, ix) => run ? null : ({ x: ix, y: island.ix })).filter(x => x),
                    borderColor: colors[island.ix % colors.length],
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
                            text: "Island"
                        }
                    }
                }
            }
        }
    )
    return errorCanvas
}

function getProblemContainer(problemName) {
    const maybeExisting = document.getElementById(`container-{problemName}`)
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

function getRunContainer(problemContainer, problemName, inputs, maxScore, timestamp) {
    const details = document.createElement("details")
    // Let's just hope these are unique.
    details.id = `run-${timestamp}`
    details.classList.add("run-container")

    const summary = document.createElement("summary")

    const problemSpan = document.createElement("span")
    problemSpan.textContent = `${problemName}(${inputs.join(', ')}) → ${maxScore}`
    summary.appendChild(problemSpan)

    const timestampLink = document.createElement("a")
    timestampLink.classList.add("timestamp")
    timestampLink.href = `#run-${timestamp}`
    timestampLink.textContent = `#${timestamp}`
    summary.appendChild(timestampLink)

    details.appendChild(summary)

    const containerInner = document.createElement("div")
    containerInner.classList.add(".details-inner")
    details.appendChild(containerInner)

    problemContainer.appendChild(details)

    return containerInner
}

function details(title, ...content) {
    const details = document.createElement("details")
    const summary = document.createElement("summary")
    summary.textContent = title

    const containerInner = document.createElement("div")
    containerInner.classList.add(".details-inner")
    details.appendChild(containerInner)

    details.appendChild(summary)
    content.forEach(elem => details.appendChild(elem))
    return details
}

function detailsCode(title, code) {
    const pre = document.createElement("pre")
    const codeElement = document.createElement("code")
    codeElement.classList.add("language-python")
    codeElement.textContent = code
    pre.appendChild(codeElement)
    hljs.highlightElement(codeElement)
    return details(title, pre)
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
            "runs": island._runs,  # noqa: SLF001
            "improvements": [(ix, str(program)) for ix, program in island._improvements],  # noqa: SLF001
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
        island.bestScore = island.runs[lastImprovement[0]]
        island.ix = i

        island.runningMaximum = island.runs.reduce((acc, score) => {
            const currentMax = score ? Math.max(acc[acc.length - 1] || 0, score) : acc[acc.length - 1] || 0
            return [...acc, currentMax]
        }, [])

        return island
    })
    islands.sort((a, b) => b.bestScore - a.bestScore)
    const maxScore = islands[0].bestScore
    const totalSuccesses = islands.reduce((acc, island) => acc + island.successCount, 0)
    const totalFailures = islands.reduce((acc, island) => acc + island.failureCount, 0)
    const totalRate = Math.round(100 * totalSuccesses / (totalSuccesses + totalFailures))

    const runContainer = getRunContainer(problemContainer, problemName, database.inputs, maxScore, database.timestamp)

    const messageSpan = document.createElement("span")
    messageSpan.textContent = database.message
    messageSpan.classList.add("message")
    runContainer.appendChild(messageSpan)

    runContainer.appendChild(detailsCode("Spec", database.specCode))

    runContainer.appendChild(details("Best Programs", ...islands.map(island => detailsCode(`Score ${island.bestScore}, Island ${island.ix}`, island.improvements[island.improvements.length - 1][1])
    )))


    runContainer.appendChild(details("Improvements over Time", improvementCanvas(islands),
        ...islands.map(island => details(`Island ${island.ix}`,
            ...island.improvements.toReversed().map(improvement => detailsCode(`Run ${improvement[0]}, Score ${island.runs[improvement[0]]}`, improvement[1])))
        )))

    runContainer.appendChild(details(`Error-rates (Total ${totalRate}% = ${totalSuccesses}/${totalSuccesses + totalFailures})`, errorCanvas(islands)))


    detailsCode(runContainer, "Config", Object.entries(database.config).map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join("\n"))
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