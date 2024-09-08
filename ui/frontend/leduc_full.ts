import { LitElement, PropertyValueMap, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { io } from 'socket.io-client'
import { createRef, Ref, ref } from 'lit/directives/ref.js';
import * as echarts from 'echarts';
import { assert, assert_tag_is_never, gap } from './utils';

type Option<t> = ["Some",t] | ["None",[]]
type Card = ["King",[]] | ["Queen",[]] | ["Jack",[]]
const card : Card[] = [["King",[]], ["Queen",[]], ["Jack",[]]]
type Action = ["Raise",[]] | ["Call",[]] | ["Fold",[]]
type Players = ["Computer",[]] | ["Human",[]] | ["Random",[]]
const possible_player_types : Players[] = [["Computer",[]], ["Human",[]], ["Random",[]]]
const players : Players[] = [possible_player_types[2],possible_player_types[1]]
type Table = {
    pot: [number, number]
    community_card: Option<Card>,
    pl_card: [Card, Card],
    raises_left: number,
    is_button_s_first_move: boolean,
    player_turn: number
}

type Game_Events =
    | ['StartGame', []]
    | ['PlayerChanged', Players[]]
    | ['ActionSelected', Action]
    | ["StartTrainingVsRando",[]]
    | ["StartTrainingVsSelf",[]]

type Game_State =
    | ["GameNotStarted", []]
    | ["WaitingForActionFromPlayerId", Table]
    | ["GameOver", Table]
    
type Message =
    | ["PlayerGotCard", [number, Card]]
    | ["CommunityCardIs", Card]
    | ["PlayerAction", [number, Action]]
    | ["Showdown", {winner_id : number; chips_won : number; cards_shown : [Card, Card]}]

type Leduc_Train_Label = string
type Leduc_Train_Serie = {
    name: string
    data: number[]
}
type Public_State = {
}
type UI_Effect = 
    | ["AddRewardsRando", number[][]]
    | ["AddRewardsSelf", number[][]]

type UI_State = {
    pl_type : Players[];
    ui_game_state : Game_State;
    messages : Message[];
}

class GameElement extends LitElement {
    dispatch_game_event = (detail : Game_Events) => {
        this.dispatchEvent(new CustomEvent('game', {bubbles: true, composed: true, detail}))
    }
}

@customElement('leduc-full-ui')
class Leduc_UI extends GameElement {
    @property({type: Object}) state : UI_State = {
        pl_type: players,
        ui_game_state: ["GameNotStarted", []],
        messages: []
    };

    // static styles = css`
    //     :host {
    //         padding: 0 50px;
    //         display: flex;
    //         flex-direction: column;
    //         box-sizing: border-box;
    //         height: 100%;
    //         width: 100%;
    //     }
    //     sl-input, sl-button {
    //         width: fit-content;
    //     }
    // `


    on_train_vs_rando = () => this.dispatch_game_event(["StartTrainingVsRando",[]])
    on_train_vs_self = () => this.dispatch_game_event(["StartTrainingVsSelf",[]])

    socket = io('/leduc_full')

    constructor(){
        super()
        this.socket.on('update', (x : [UI_State, UI_Effect[]]) => {
            this.state = x[0];
            this.process_effects(x[1])
        });
        this.addEventListener('full', (ev) => {
            ev.stopPropagation();
            this.socket.emit('update', (ev as CustomEvent<Game_Events>).detail);
        })
    }

    vs_rando_chart = createRef<Training_Chart>()
    vs_self_chart = createRef<Training_Chart>()
    process_effects(l: UI_Effect[]) {
        const average = (data : number[][]) => data.map(x => x.reduce((a,b) => a+b) / x.length)
        l.forEach(l => {
            const [tag, data] = l
            switch (tag) {
                case 'AddRewardsRando': {
                    console.log({
                        rando_data: data,
                        rando_average: average(data)
                    });
                    this.vs_rando_chart.value?.add_rewards(data)
                    break;
                }
                case 'AddRewardsSelf': {
                    console.log({
                        self_data: data,
                        self_average: average(data)
                    });
                    this.vs_self_chart.value?.add_rewards(data)
                    break;
                }
                default: assert_tag_is_never(tag);
            }
        })
    }

    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            height: 300%;
        }

        .play_area {
            display: flex;
            flex-direction: row;
            box-sizing: border-box;
            height: 100%;
        }
        .train_area {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding: 5px;
            border: solid 5px black;
        }

        training-full-chart {
            flex: 1;
        }

        leduc-full-menu {
            flex: 1;
        }
        
        .game_area {
            display: flex;
            flex-direction: column;
            flex: 5;
        }

        leduc-full-game {
            flex: 4;
        }
        
        leduc-full-history {
            flex: 1;
        }
    `

    render(){
        return html`
            <div class="play_area">
                <leduc-full-menu .pl_type=${this.state.pl_type}></leduc-full-menu>
                ${gap(10)}
                <div class="game_area">
                    <leduc-full-game .state=${this.state.ui_game_state}></leduc-full-game>
                    ${gap(10)}
                    <leduc-full-history .messages=${this.state.messages}></leduc-full-history>
                </div>
            </div>
            ${gap(10)}
            <div class="train_area">
                <training-full-chart ${ref(this.vs_rando_chart)}></training-full-chart>
                <br/>
                <sl-button variant="primary" @click=${this.on_train_vs_rando}>Train (vs Random)</sl-button>
            </div>
            ${gap(10)}
            <div class="train_area">
                <training-full-chart ${ref(this.vs_self_chart)}></training-full-chart>
                <br/>
                <sl-button variant="primary" @click=${this.on_train_vs_self}>Train (vs Self)</sl-button>
            </div>
        `
    }
}

@customElement('leduc-full-menu')
class Leduc_Menu extends GameElement {
    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            background-color: hsl(0,100%,98%);
            padding: var(--sl-spacing-x-small);
            border: 3px solid;
            border-radius: 5px;
            height: 100%;
            align-items: center;
        }

        div {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        sl-select {
            text-align: center
        }
    `

    @property({type: Array}) pl_type : Players[] = players;
    
    start_game = () => this.dispatch_game_event(['StartGame', []])
    on_change = (pl_id : number) => (ev : any) => {
        const find_player = () => {
            const pl_name : string = ev.target.value
            for (const pl of players) {
                if (pl[0] === pl_name) {
                    return pl;
                }
            }
            throw Error("Cannot find the player.")
        }
        const pl_type = this.pl_type.map((x,i) => i !== pl_id ? x : find_player());
        this.dispatch_game_event(["PlayerChanged", pl_type])
    }

    render() {
        return html`
            <div>
                <sl-button @click=${this.start_game}>Start Game</sl-button>
            </div>
            ${gap(20)}
            <div>
                <sl-select name="pl1" id="pl1" .value=${this.pl_type[0][0]} @sl-change=${this.on_change(0)}>
                    <div slot="label">Player 0:</div>
                    ${possible_player_types.map(([x,[]]) =>
                        html`<sl-option value=${x}>${x}</sl-option>`
                    )}
                </sl-select>
            </div>
            ${gap(20)}
            <div>
                <sl-select name="pl2" id="pl2" .value=${this.pl_type[1][0]} @sl-change=${this.on_change(1)}>
                    <div slot="label">Player 1:</div>
                    ${possible_player_types.map(([x,[]]) =>
                        html`<sl-option value=${x}>${x}</sl-option>`
                    )}
                </sl-select>
            </div>
            `
    }
}

@customElement('leduc-full-history')
class Leduc_History extends GameElement {
    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            background-color: white;
            padding: 4px;
            border: 3px solid;
            border-radius: 5px;
            overflow: auto;
            font-family: var(--sl-font-mono);
        }

        div {
            color: gray;
        }
    `

    @property({type: Array}) messages : Message[] = []

    protected updated(_changedProperties: PropertyValueMap<any> | Map<PropertyKey, unknown>): void {
        // Scroll the message window to the bottom on ever new message.
        this.scrollTo({top: this.scrollHeight});
    }

    print_card = (x : Card) => x[0]
    print_action = (x : Action) => {
        switch (x[0]) {
            case 'Raise': return "raises"
            case 'Call': return "calls"
            case 'Fold': return "folds"
        }
    }

    print_message = (x : Message) : string[] => {
        const [tag,arg] = x
        switch (tag) {
            case 'PlayerGotCard': {
                return [`Player ${arg[0]} got ${this.print_card(arg[1])}`]
            }
            case 'CommunityCardIs': {
                return [`The community card is ${this.print_card(arg)}`]
            }
            case 'PlayerAction': {
                return [`Player ${arg[0]} ${this.print_action(arg[1])}.`]
            }
            case 'Showdown': {
                const {winner_id, chips_won, cards_shown} = arg
                return [
                    `Player 0 shows ${this.print_card(cards_shown[0])}.`,
                    `Player 1 shows ${this.print_card(cards_shown[1])}.`,
                    winner_id === -1
                    ? "The game is a tie."
                    : `Player ${winner_id} wins ${chips_won} chips!`,
                    "The game is over."
                ]
            }
            default : return assert_tag_is_never(tag)
        }
    }
    
    render() {
        return html`
            ${map((this.messages).flatMap(this.print_message), x => html`
                <div>${x}</div>
            `)}
        `
    }
}

@customElement('leduc-full-pot')
class Leduc_Pot extends LitElement {
    @property({type: Number}) pot = 0;

    static styles = css`
        :host {
            display: block;
            border: 2px dashed black;
            width: fit-content;
            font-size: 2em;
            padding: 5px;
        }
    ` 

    render() {
        return html`${this.pot}`
    }
}

@customElement('leduc-full-card')
class Leduc_Card extends LitElement {
    @property({type: Array}) card : Option<Card> = ["Some", card[0]];
    @property({type: Boolean}) card_visible = true;

    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 4px solid black;
            height: fit-content;
            width: fit-content;
            padding: 10px;
            background-color: burlywood;
            user-select: none;
            text-align: center;
            font-family: var(--sl-font-mono);
            font-size: var(--sl-font-size-2x-large);
        }

        div {
            white-space: pre;
        }
    `

    render(){
        return html`<div>${this.card_visible && this.card[0] === "Some" ? this.card[1][0][0] : " "}</div>`
    }
}

@customElement('leduc-full-game')
class Leduc_Game extends GameElement {
    @property({type: Array}) state : Game_State = ["WaitingForActionFromPlayerId", {
        community_card: ["None",[]],
        pl_card: [card[0], card[1]],
        pot: [4,3],
        is_button_s_first_move: true,
        player_turn: 0,
        raises_left: 0
    }]

    static styles = css`
        :host {
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            background-color: white;
            padding: 20px;
            border: 3px solid;
            border-radius: 5px;
            align-items: center;
            justify-content: space-between
        }
        
        .row {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }

        .flex-1 {
            flex: 1;
        }

        .flex-pot {
            display: flex;
            flex: 1;
            justify-content: flex-end;
        }

        .flex-card {
            display: flex;
            flex-basis: 200px;
            justify-content: center;
        }
        
        .flex-actions {
            display: flex;
            flex: 1;
            flex-direction: row;
            justify-content: flex-start;
        }

        button {
            font-size: inherit;
        }
        `

    on_action = (action : Action) => () => {
        this.dispatch_game_event(["ActionSelected", action])
    }

    render_state(){
        const [tag,arg] = this.state
        const some = (x : Card) : Option<Card> => ["Some", x]
        const f = (is_current : boolean, card_visible : boolean, id : number, table : Table) => {
            return html`
                <div class="row">
                    <div class="flex-pot">
                        <leduc-full-pot .pot=${table.pot[id]}></leduc-full-pot>
                    </div>
                    <div class="flex-card">
                        <leduc-full-card .card=${some(table.pl_card[id])} ?card_visible=${card_visible}></leduc-full-card>
                    </div>
                    ${
                        is_current
                        ? html`
                            <div class="flex-actions">
                                <sl-button ?disabled=${table.pot[0] === table.pot[1]} @click=${this.on_action(["Fold",[]])}>Fold</sl-button>
                                <sl-button @click=${this.on_action(["Call",[]])}>Call</sl-button>
                                <sl-button ?disabled=${!(table.raises_left > 0)} @click=${this.on_action(["Raise",[]])}>Raise</sl-button>
                            </div>
                            `
                        : html`
                            <div class="flex-1"></div>
                        `
                    }
                </div>
            `}
        
        switch (tag){
            case "GameNotStarted": return html`
                <div>
                    Please start the game...
                </div>
            `
            case "WaitingForActionFromPlayerId":{ 
                const table = arg;
                const f_ = (c : number) => f(table.player_turn === c, table.player_turn === c, c, table)
                return html`
                    ${f_(0)}
                    <div>
                        <leduc-full-card .card=${table.community_card}></leduc-full-card>
                    </div>
                    ${f_(1)}
                    `
                }
            case "GameOver": {
                const table = arg;
                return html`
                    ${f(false, true, 0, table)}
                    <div>
                        <leduc-full-card .card=${table.community_card}></leduc-full-card>
                    </div>
                    ${f(false, true, 1, table)}
                `
            }
        }
    }

    render() {
        return html`${this.render_state()}`
    }
}

const assert_chart_data = (labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) => {
    assert(series.every(x => labels.length === x.data.length), "The length of the labels array does not match that of the series data.");
}
const update_chart = (chart: echarts.ECharts, labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) => {
    assert_chart_data(labels, series);
    chart.setOption({
        xAxis: { data: labels },
        series: series.map(x => ({ ...x, type: "line" }))
    })
}

@customElement('training-full-chart')
class Training_Chart extends LitElement {
    static styles = css`
        :host {
            display: block;
            height: 100%;
            width: 100%;
        }
    `

    labels: Leduc_Train_Label[] = []
    series: Leduc_Train_Serie[] = []

    chart?: echarts.ECharts;

    add_rewards(rewards: number[][]) {
        const labels : Leduc_Train_Label[] = []
        for (let i = 0; i < rewards[0].length; i++) {
            labels.push((this.labels.length + i).toString())
        }
        this.add_item(labels, rewards.map((data,i) => ({ name: i.toString(), data })))
    }

    add_item(labels: Leduc_Train_Label[], series: Leduc_Train_Serie[]) {
        assert_chart_data(labels, series);
        if (
            this.series.length === series.length
            && this.series.every((x, i) => x.name === series[i].name)
        ) {
            this.labels.push(...labels);
            this.series.forEach((x, i) => x.data.push(...series[i].data))
        } else {
            this.labels = labels;
            this.series = series;
        }
        this.update_chart();
    }

    render() {
        // The slot will put the chart which is in the light DOM into the shadow root.
        return html`<slot></slot>`;
    }

    update_chart() {
        if (this.chart) {
            update_chart(this.chart, this.labels, this.series);
        }
    }

    firstUpdated() {
        // Create the echarts instance
        this.chart = echarts.init(this);

        // Draw the chart
        this.chart.setOption({
            title: {
                text: 'RL Agent Training Run'
            },
            yAxis: {
                boundaryGap: [0, '50%'],
                type: 'value'
            },
            tooltip: {},
            xAxis: { data: [] },
        });

        // Without this the chart sizing wouldn't work properly.
        const chart_container_resize_observer = new ResizeObserver(() => this.chart?.resize());
        chart_container_resize_observer.observe(this)
    }
}
