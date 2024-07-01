DROP TABLE IF EXISTS Stock_Investments_90;
CREATE TABLE Stock_Investments_90(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_45;
CREATE TABLE Stock_Investments_45(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_30;
CREATE TABLE Stock_Investments_30(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_15;
CREATE TABLE Stock_Investments_15(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_9;
CREATE TABLE Stock_Investments_9(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_3;
CREATE TABLE Stock_Investments_3(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);


DROP TABLE IF EXISTS Stock_Investments_1;
CREATE TABLE Stock_Investments_1(
    id int not null unique,
    ticker varchar(10),
    sell_after int,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_half;
CREATE TABLE Stock_Investments_half(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_1x;
CREATE TABLE Stock_Investments_Volatility_1x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_2x;
CREATE TABLE Stock_Investments_Volatility_2x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_5x;
CREATE TABLE Stock_Investments_Volatility_5x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);


DROP TABLE IF EXISTS Stock_Investments_Volatility_8x;
CREATE TABLE Stock_Investments_Volatility_8x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_10x;
CREATE TABLE Stock_Investments_Volatility_10x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_13x;
CREATE TABLE Stock_Investments_Volatility_13x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_17x;
CREATE TABLE Stock_Investments_Volatility_17x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Volatility_20x;
CREATE TABLE Stock_Investments_Volatility_20x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_halfx;
CREATE TABLE Stock_Investments_Volatility_Lambda_halfx(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_1x;
CREATE TABLE Stock_Investments_Volatility_Lambda_1x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_2x;
CREATE TABLE Stock_Investments_Volatility_Lambda_2x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_3x;
CREATE TABLE Stock_Investments_Volatility_Lambda_3x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_4x;
CREATE TABLE Stock_Investments_Volatility_Lambda_4x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Stock_Investments_Lambda_5x;
CREATE TABLE Stock_Investments_Volatility_Lambda_5x(
    id int not null unique,
    ticker varchar(10),
    sell_after float,
    price float,
    volatility float,
    volatility_coeff float,
    drift float
);

DROP TABLE IF EXISTS Lineitem_20000;
CREATE TABLE Lineitem_20000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_60000;
CREATE TABLE Lineitem_60000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_120000;
CREATE TABLE Lineitem_120000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_300000;
CREATE TABLE Lineitem_300000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_450000;
CREATE TABLE Lineitem_450000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_600000;
CREATE TABLE Lineitem_600000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_1200000;
CREATE TABLE Lineitem_1200000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_3000000;
CREATE TABLE Lineitem_3000000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_4500000;
CREATE TABLE Lineitem_4500000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_6000000;
CREATE TABLE Lineitem_6000000(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_1x;
CREATE TABLE Lineitem_Variance_1x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_2x;
CREATE TABLE Lineitem_Variance_2x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_5x;
CREATE TABLE Lineitem_Variance_5x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_Variance_8x;
CREATE TABLE Lineitem_Variance_8x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_10x;
CREATE TABLE Lineitem_Variance_10x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_13x;
CREATE TABLE Lineitem_Variance_13x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Variance_17x;
CREATE TABLE Lineitem_Variance_17x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_Variance_20x;
CREATE TABLE Lineitem_Variance_20x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_Lambda_halfx;
CREATE TABLE Lineitem_Lambda_halfx(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Lambda_1x;
CREATE TABLE Lineitem_Lambda_1x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Lambda_2x;
CREATE TABLE Lineitem_Lambda_2x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);


DROP TABLE IF EXISTS Lineitem_Lambda_3x;
CREATE TABLE Lineitem_Lambda_3x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Lambda_4x;
CREATE TABLE Lineitem_Lambda_4x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);

DROP TABLE IF EXISTS Lineitem_Lambda_5x;
CREATE TABLE Lineitem_Lambda_5x(
    id int not null unique,
    orderkey int not null,
    partkey int not null,
    suppkey int not null,
    linenumber int,
    quantity float,
    quantity_mean float,
    quantity_variance float,
    quantity_variance_coeff float,
    price float,
    price_mean float,
    price_variance float,
    price_variance_coeff float,
    tax float
);